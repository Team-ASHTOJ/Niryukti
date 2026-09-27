#include "internal.hpp"
#include <climits>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
namespace vantage {
namespace {
void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void sparse_check(cusparseStatus_t e, const char *what) {
    if (e != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuSPARSE status " + std::to_string(int(e)));
}
template <class T> struct Buffer {
    T *p = nullptr;
    size_t n = 0;
    Buffer() = default;
    explicit Buffer(size_t n_) : n(n_) {
        if (n)
            check(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)), "cudaMalloc");
    }
    explicit Buffer(const std::vector<T> &v) : Buffer(v.size()) {
        upload(v);
    }
    ~Buffer() {
        if (p)
            cudaFree(p);
    }
    Buffer(Buffer &&other) noexcept : p(other.p), n(other.n) {
        other.p = nullptr;
        other.n = 0;
    }
    Buffer &operator=(Buffer &&other) noexcept {
        if (this != &other) {
            if (p)
                cudaFree(p);
            p = other.p;
            n = other.n;
            other.p = nullptr;
            other.n = 0;
        }
        return *this;
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void upload(const std::vector<T> &v) {
        if (v.size() != n)
            throw std::runtime_error("GPU upload size mismatch");
        if (n)
            check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice), "H2D copy");
    }
    void download(std::vector<T> &v) {
        v.resize(n);
        if (n)
            check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "D2H copy");
    }
    void zero() {
        if (n)
            check(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
    }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() {
        check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream");
    }
    ~Stream() {
        if (value)
            cudaStreamDestroy(value);
    }
};
struct Handle {
    cusparseHandle_t h = nullptr;
    Handle() {
        sparse_check(cusparseCreate(&h), "cusparseCreate");
    }
    ~Handle() {
        if (h)
            cusparseDestroy(h);
    }
};
struct Matrix {
    std::unique_ptr<Buffer<int64_t>> ptr64, index64;
    std::unique_ptr<Buffer<int32_t>> ptr32, index32;
    std::unique_ptr<Buffer<double>> values64;
    std::unique_ptr<Buffer<float>> values32;
    size_t nnz;
    cusparseSpMatDescr_t d = nullptr;
    explicit Matrix(const Sparse &s, const std::string &indices, bool mixed) : nnz(s.value.size()) {
        bool narrow = s.rows <= INT32_MAX && s.cols <= INT32_MAX && nnz <= INT32_MAX;
        if (indices == "32" && !narrow)
            throw std::runtime_error("Model exceeds 32-bit CSR limits");
        narrow = narrow && indices != "64";
        void *ptr, *index, *values;
        if (narrow) {
            std::vector<int32_t> p(s.ptr.begin(), s.ptr.end()), i(s.index.begin(), s.index.end());
            ptr32 = std::make_unique<Buffer<int32_t>>(p);
            index32 = std::make_unique<Buffer<int32_t>>(i);
            ptr = ptr32->p;
            index = index32->p;
        } else {
            ptr64 = std::make_unique<Buffer<int64_t>>(s.ptr);
            index64 = std::make_unique<Buffer<int64_t>>(s.index);
            ptr = ptr64->p;
            index = index64->p;
        }
        if (mixed) {
            std::vector<float> coefficients;
            for (double v : s.value) {
                float f = float(v);
                if (!std::isfinite(f) || (v != 0 && f == 0))
                    throw std::runtime_error(
                        "FP32 matrix conversion would overflow or erase a coefficient; use fp64");
                coefficients.push_back(f);
            }
            values32 = std::make_unique<Buffer<float>>(coefficients);
            values = values32->p;
        } else {
            values64 = std::make_unique<Buffer<double>>(s.value);
            values = values64->p;
        }
        sparse_check(cusparseCreateCsr(&d, s.rows, s.cols, nnz, ptr, index, values,
                                       narrow ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I,
                                       narrow ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I,
                                       CUSPARSE_INDEX_BASE_ZERO, mixed ? CUDA_R_32F : CUDA_R_64F),
                     "create CSR");
    }
    ~Matrix() {
        if (d)
            cusparseDestroySpMat(d);
    }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        clear();
    }
    void clear() {
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
        executable = nullptr;
        graph = nullptr;
    }
};
struct Vector {
    cusparseDnVecDescr_t d = nullptr;
    Vector(size_t n, double *p) {
        sparse_check(cusparseCreateDnVec(&d, n, p, CUDA_R_64F), "create vector");
    }
    ~Vector() {
        if (d)
            cusparseDestroyDnVec(d);
    }
};

// State: step factor, averaging mass, accepted count, rejected count, accept flag, average weight.
// Trial decisions and reductions stay on the device; only six scalars return per monitoring chunk.
__device__ double block_sum(double value) {
    __shared__ double scratch[256];
    int t = threadIdx.x;
    scratch[t] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride)
            scratch[t] += scratch[t + stride];
        __syncthreads();
    }
    double answer = scratch[0];
    __syncthreads();
    return answer;
}
__global__ void primal_trial(int64_t n, const double *x, double *trial, double *xb,
                             const double *aty, const double *c, const double *q, const double *qx,
                             bool full_q, const double *l, const double *u, double base_tau,
                             const double *state, double *partial) {
    auto j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    double square = 0;
    if (j < n) {
        double tau = base_tau * state[0];
        double v = fmin(
            u[j], fmax(l[j], (x[j] - tau * (c[j] + aty[j] + (full_q ? qx[j] + q[j] * x[j] : 0))) /
                                 (1 + tau * (full_q ? 0 : q[j]))));
        trial[j] = v;
        xb[j] = v - x[j]; // A*dx is more stable than (A*xbar - A*x)/2.
        square = (v - x[j]) * (v - x[j]);
        if (!isfinite(v))
            square = INFINITY;
    }
    double sum = block_sum(square);
    if (!threadIdx.x)
        partial[blockIdx.x] = sum;
}
__global__ void dual_trial(int64_t n, const double *y, double *trial, const double *delta,
                           const double *previous_ax, const double *l, const double *u,
                           double base_sigma, const double *state, double *partial,
                           double *couplings) {
    auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    double square = 0, coupling = 0;
    if (i < n) {
        double sigma = base_sigma * state[0];
        double bar = previous_ax[i] + 2 * delta[i];
        double z = y[i] + sigma * bar;
        double v = fmax(0., z - sigma * u[i]) + fmin(0., z - sigma * l[i]);
        trial[i] = v;
        double dy = v - y[i];
        square = dy * dy;
        coupling = dy * delta[i];
        if (!isfinite(v) || !isfinite(bar) || !isfinite(delta[i]))
            square = INFINITY;
    }
    double s = block_sum(square), c = block_sum(coupling);
    if (!threadIdx.x) {
        partial[blockIdx.x] = s;
        couplings[blockIdx.x] = c;
    }
}
__global__ void decide(int nx, int ny, const double *dx, const double *dy, const double *cross,
                       double tau, double sigma, bool adaptive, bool halpern,
                       bool residual_restarts, double *state) {
    double p = 0, d = 0, c = 0;
    for (int i = threadIdx.x; i < nx; i += 256)
        p += dx[i];
    for (int i = threadIdx.x; i < ny; i += 256) {
        d += dy[i];
        c += cross[i];
    }
    p = block_sum(p);
    d = block_sum(d);
    c = block_sum(c);
    if (!threadIdx.x) {
        if (state[9]) {
            state[4] = 0;
            return;
        }
        double factor = state[0];
        double energy = p / (tau * factor) + d / (sigma * factor);
        double limit = fabs(c) > 0 ? energy / (2 * fabs(c)) : INFINITY;
        bool finite = isfinite(energy) && isfinite(c);
        bool accept = finite && (!adaptive || limit >= 1);
        if (adaptive) {
            double k = state[2] + 2;
            double multiplier = fmin((1 - pow(k, -.3)) * limit, 1 + pow(k, -.6));
            if (!finite || !isfinite(multiplier) || multiplier <= 0)
                multiplier = .5;
            state[0] = fmin(1e12, fmax(1e-12, factor * multiplier));
        }
        state[4] = accept;
        if (accept) {
            state[1] += factor;
            state[2] += 1;
            state[5] = factor / state[1];
            if (halpern) {
                double residual = sqrt(fmax(0., energy - 2 * c));
                if (state[6] == 0)
                    state[7] = residual;
                state[10] = 1 / (state[6] + 2);
                state[6] += 1;
                state[9] = residual_restarts && state[6] >= 10 &&
                           (residual <= .2 * state[7] ||
                            (residual <= .8 * state[7] && residual > state[8]) ||
                            state[6] >= fmax(10., state[2] / 2));
                state[8] = residual;
            }
        } else
            state[3] += 1;
    }
}
__global__ void commit_primal(int64_t n, double *x, const double *trial, double *avg,
                              const double *state, const double *anchor, bool halpern,
                              double reflection) {
    auto j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < n && state[4]) {
        double old = x[j];
        if (halpern) {
            avg[j] = trial[j];
            x[j] = state[10] * anchor[j] +
                   (1 - state[10]) * ((1 + reflection) * trial[j] - reflection * old);
        } else {
            x[j] = trial[j];
            avg[j] += (x[j] - avg[j]) * state[5];
        }
    }
}
__global__ void commit_dual(int64_t n, double *y, const double *trial, double *avg,
                            double *previous_ax, const double *delta, const double *state,
                            const double *anchor, const double *anchor_ax, bool halpern,
                            double reflection) {
    auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && state[4]) {
        if (halpern) {
            avg[i] = trial[i];
            y[i] = state[10] * anchor[i] +
                   (1 - state[10]) * ((1 + reflection) * trial[i] - reflection * y[i]);
            previous_ax[i] = state[10] * anchor_ax[i] +
                             (1 - state[10]) * (previous_ax[i] + (1 + reflection) * delta[i]);
        } else {
            y[i] = trial[i];
            avg[i] += (y[i] - avg[i]) * state[5];
            previous_ax[i] += delta[i];
        }
    }
}
// Scaled-space control diagnostics. This never certifies optimality; the
// independent original-model verifier remains the final acceptance authority.
__device__ double block_max(double value) {
    __shared__ double scratch[256];
    int t = threadIdx.x;
    scratch[t] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride)
            scratch[t] = fmax(scratch[t], scratch[t + stride]);
        __syncthreads();
    }
    double result = scratch[0];
    __syncthreads();
    return result;
}
__global__ void monitor_partial(int64_t n, int64_t rows, const double *x, const double *y,
                                const double *ax, const double *aty, const double *qx,
                                const double *c, const double *q, const double *lb,
                                const double *ub, const double *rl, const double *ru, double cscale,
                                double *partial) {
    auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    double primal = 0, dual = 0, objective = 0, complement = 0, bad = 0;
    if (i < n) {
        double g = c[i] + q[i] * x[i] + aty[i] + (qx ? qx[i] : 0);
        primal = fmax(0., fmax(lb[i] - x[i], x[i] - ub[i])) / (1 + fabs(x[i]));
        dual = fabs(fmin(x[i] - lb[i], fmax(x[i] - ub[i], g))) / cscale;
        objective = c[i] * x[i] + .5 * q[i] * x[i] * x[i] + (qx ? .5 * qx[i] * x[i] : 0);
        if (g > 0 && isfinite(lb[i]))
            complement += fabs(g * (x[i] - lb[i]));
        if (g < 0 && isfinite(ub[i]))
            complement += fabs(g * (ub[i] - x[i]));
        bad = (!isfinite(x[i]) || !isfinite(g) || !isfinite(objective));
    }
    if (i < rows) {
        if (isfinite(rl[i]))
            primal = fmax(primal, fmax(0., rl[i] - ax[i]) / (1 + fabs(rl[i])));
        if (isfinite(ru[i]))
            primal = fmax(primal, fmax(0., ax[i] - ru[i]) / (1 + fabs(ru[i])));
        if (y[i] != 0) {
            double endpoint = y[i] > 0 ? ru[i] : rl[i];
            if (isfinite(endpoint))
                complement += fabs(y[i] * (endpoint - ax[i]));
            else
                dual = fmax(dual, fabs(y[i]));
        }
        bad = fmax(bad, double(!isfinite(ax[i]) || !isfinite(y[i])));
    }
    double p = block_max(primal), d = block_max(dual), o = block_sum(objective),
           v = block_sum(complement), b = block_max(bad);
    if (threadIdx.x == 0) {
        auto k = 5 * blockIdx.x;
        partial[k] = p;
        partial[k + 1] = d;
        partial[k + 2] = o;
        partial[k + 3] = v;
        partial[k + 4] = b;
    }
}
__global__ void monitor_reduce(int64_t count, const double *partial, double *result) {
    double p = 0, d = 0, o = 0, v = 0, b = 0;
    for (int64_t k = threadIdx.x; k < count; k += blockDim.x) {
        p = fmax(p, partial[5 * k]);
        d = fmax(d, partial[5 * k + 1]);
        o += partial[5 * k + 2];
        v += partial[5 * k + 3];
        b = fmax(b, partial[5 * k + 4]);
    }
    p = block_max(p);
    d = block_max(d);
    o = block_sum(o);
    v = block_sum(v);
    b = block_max(b);
    if (threadIdx.x == 0)
        result[0] = b ? INFINITY : fmax(fmax(p, d), v / (1 + fabs(o)));
}
class CudaBackend final : public IterationBackend {
    Stream stream;
    Handle handle;
    Matrix a, at;
    std::unique_ptr<Matrix> quadratic;
    std::unique_ptr<Buffer<double>> quadratic_product;
    std::unique_ptr<Vector> q_input, q_output;
    double a_norm = 0, q_norm = 0;
    Buffer<double> x, y, xb, xa, ya, ax, aty, c, q, lb, ub, rl, ru;
    Buffer<double> trial_x, trial_y, previous_ax, dx, dy, cross, state;
    Buffer<double> anchor_x, anchor_y, anchor_ax;
    bool adaptive, halpern, residual_restarts, graphs;
    double reflection, graph_tau = 0, graph_sigma = 0;
    Graph graph;
    bool request_restart = false;
    int64_t accepted = 0, rejected = 0;
    int empty_chunks = 0;
    Vector vx, vy, vax, vaty;
    std::unique_ptr<Buffer<char>> workspace;
    std::unique_ptr<Buffer<char>> monitor_workspace;
    Buffer<double> monitor_ax, monitor_aty, monitor_qx, monitor_partials, monitor_scalar;
    std::unique_ptr<Vector> monitor_x, monitor_y, monitor_aout, monitor_atout, monitor_qout;
    double monitor_cscale = 1;
    bool monitor_initialized = false;

  public:
    CudaBackend(const Model &m, const std::vector<double> &px, const std::vector<double> &py,
                bool adapt, bool anchored, double reflect, bool residual_restart, bool graph_mode,
                const std::string &indices, const std::string &precision)
        : a(m.A, indices, precision == "mixed"), at(m.A.transpose(), indices, precision == "mixed"),
          x(px), y(py), xb(px), xa(px.size()), ya(py.size()), ax(py.size()), aty(px.size()), c(m.c),
          q(m.q), lb(m.lb), ub(m.ub), rl(m.rl), ru(m.ru), trial_x(px.size()), trial_y(py.size()),
          previous_ax(m.A.multiply(px)), dx((px.size() + 255) / 256), dy((py.size() + 255) / 256),
          cross((py.size() + 255) / 256),
          state(std::vector<double>{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
          anchor_x(anchored ? px : std::vector<double>{}),
          anchor_y(anchored ? py : std::vector<double>{}),
          anchor_ax(anchored ? m.A.multiply(px) : std::vector<double>{}), adaptive(adapt),
          halpern(anchored), residual_restarts(residual_restart), graphs(graph_mode),
          reflection(reflect), vx(xb.n, xb.p), vy(y.n, y.p), vax(ax.n, ax.p), vaty(aty.n, aty.p) {
        sparse_check(cusparseSetStream(handle.h, stream.value), "set sparse stream");
        if (!m.Q.value.empty()) {
            if (precision != "fp64")
                throw std::runtime_error("Sparse QP requires FP64 matrices");
            adaptive = false;
            q_norm = m.quadratic_norm_bound();
            quadratic = std::make_unique<Matrix>(m.Q, indices, false);
            quadratic_product = std::make_unique<Buffer<double>>(m.c.size());
            q_input = std::make_unique<Vector>(x.n, x.p);
            q_output = std::make_unique<Vector>(x.n, quadratic_product->p);
            std::vector<double> cols(m.c.size());
            double rowmax = 0;
            for (int64_t i = 0; i < m.A.rows; ++i) {
                double sum = 0;
                for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k) {
                    sum += std::abs(m.A.value[k]);
                    cols[m.A.index[k]] += std::abs(m.A.value[k]);
                }
                rowmax = std::max(rowmax, sum);
            }
            double colmax = 0;
            for (auto v : cols)
                colmax = std::max(colmax, v);
            a_norm = std::sqrt(rowmax) * std::sqrt(colmax);
        }
        xa.zero();
        ya.zero();
        ax.zero();
        aty.zero();
        double one = 1, zero = 0;
        size_t s1 = 0, s2 = 0;
        if (m.A.value.size()) {
            sparse_check(cusparseSpMV_bufferSize(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                                 a.d, vx.d, &zero, vax.d, CUDA_R_64F,
                                                 CUSPARSE_SPMV_CSR_ALG2, &s1),
                         "A*x buffer size");
            sparse_check(cusparseSpMV_bufferSize(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                                 at.d, vy.d, &zero, vaty.d, CUDA_R_64F,
                                                 CUSPARSE_SPMV_CSR_ALG2, &s2),
                         "At*y buffer size");
        }
        size_t sq = 0;
        if (quadratic)
            sparse_check(cusparseSpMV_bufferSize(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                                 quadratic->d, q_input->d, &zero, q_output->d,
                                                 CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &sq),
                         "Q*x workspace");
        workspace = std::make_unique<Buffer<char>>(std::max({s1, s2, sq}));
        check(cudaDeviceSynchronize(), "backend initialization sync");
    }

    double monitor_point(double *px, double *py) {
        sparse_check(cusparseDnVecSetValues(monitor_x->d, px), "monitor x pointer");
        sparse_check(cusparseDnVecSetValues(monitor_y->d, py), "monitor y pointer");
        double one = 1, zero = 0;
        if (a.nnz) {
            sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a.d,
                                      monitor_x->d, &zero, monitor_aout->d, CUDA_R_64F,
                                      CUSPARSE_SPMV_CSR_ALG2, monitor_workspace->p),
                         "monitor A*x");
            sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, at.d,
                                      monitor_y->d, &zero, monitor_atout->d, CUDA_R_64F,
                                      CUSPARSE_SPMV_CSR_ALG2, monitor_workspace->p),
                         "monitor At*y");
        }
        if (quadratic)
            sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                      quadratic->d, monitor_x->d, &zero, monitor_qout->d,
                                      CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, monitor_workspace->p),
                         "monitor Q*x");
        auto blocks = (std::max(x.n, y.n) + 255) / 256;
        if (!blocks)
            return 0;
        monitor_partial<<<blocks, 256, 0, stream.value>>>(
            x.n, y.n, px, py, monitor_ax.p, monitor_aty.p, quadratic ? monitor_qx.p : nullptr, c.p,
            q.p, lb.p, ub.p, rl.p, ru.p, monitor_cscale, monitor_partials.p);
        monitor_reduce<<<1, 256, 0, stream.value>>>(blocks, monitor_partials.p, monitor_scalar.p);
        check(cudaGetLastError(), "monitor reduction");
        check(cudaStreamSynchronize(stream.value), "monitor sync");
        std::vector<double> scalar;
        monitor_scalar.download(scalar);
        return scalar[0];
    }
    double monitor() override {
        if (!monitor_initialized) {
            monitor_ax = Buffer<double>(y.n);
            monitor_aty = Buffer<double>(x.n);
            monitor_qx = Buffer<double>(quadratic ? x.n : 0);
            monitor_partials = Buffer<double>(5 * ((std::max(x.n, y.n) + 255) / 256));
            monitor_scalar = Buffer<double>(1);
            monitor_ax.zero();
            monitor_aty.zero();
            monitor_qx.zero();
            monitor_x = std::make_unique<Vector>(x.n, x.p);
            monitor_y = std::make_unique<Vector>(y.n, y.p);
            monitor_aout = std::make_unique<Vector>(y.n, monitor_ax.p);
            monitor_atout = std::make_unique<Vector>(x.n, monitor_aty.p);
            if (quadratic)
                monitor_qout = std::make_unique<Vector>(x.n, monitor_qx.p);
            double one = 1, zero = 0;
            size_t s1 = 0, s2 = 0, sq = 0;
            if (a.nnz) {
                sparse_check(cusparseSpMV_bufferSize(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &one, a.d, monitor_x->d, &zero,
                                                     monitor_aout->d, CUDA_R_64F,
                                                     CUSPARSE_SPMV_CSR_ALG2, &s1),
                             "monitor A workspace");
                sparse_check(cusparseSpMV_bufferSize(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &one, at.d, monitor_y->d, &zero,
                                                     monitor_atout->d, CUDA_R_64F,
                                                     CUSPARSE_SPMV_CSR_ALG2, &s2),
                             "monitor At workspace");
            }
            if (quadratic)
                sparse_check(cusparseSpMV_bufferSize(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &one, quadratic->d, monitor_x->d, &zero,
                                                     monitor_qout->d, CUDA_R_64F,
                                                     CUSPARSE_SPMV_CSR_ALG2, &sq),
                             "monitor Q workspace");
            monitor_workspace = std::make_unique<Buffer<char>>(std::max({s1, s2, sq}));
            std::vector<double> cost;
            c.download(cost);
            for (auto v : cost)
                monitor_cscale = std::max(monitor_cscale, 1 + std::abs(v));
            check(cudaDeviceSynchronize(), "monitor setup sync");
            monitor_initialized = true;
        }
        return std::min(monitor_point(x.p, y.p), monitor_point(xa.p, ya.p));
    }
    int64_t rejected_steps() const override {
        return rejected;
    }
    bool restart_requested() const override {
        return request_restart;
    }
    void launch_iteration(double tau, double sigma) {
        double one = 1, zero = 0;
        if (a.nnz)
            sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, at.d, vy.d,
                                      &zero, vaty.d, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                      workspace->p),
                         "SpMV At*y");
        if (quadratic)
            sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                      quadratic->d, q_input->d, &zero, q_output->d, CUDA_R_64F,
                                      CUSPARSE_SPMV_CSR_ALG2, workspace->p),
                         "Q*x");
        if (x.n)
            primal_trial<<<dx.n, 256, 0, stream.value>>>(
                x.n, x.p, trial_x.p, xb.p, aty.p, c.p, q.p,
                quadratic_product ? quadratic_product->p : nullptr, bool(quadratic), lb.p, ub.p,
                tau, state.p, dx.p);
        check(cudaGetLastError(), "primal trial kernel");
        if (a.nnz)
            sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a.d, vx.d,
                                      &zero, vax.d, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                      workspace->p),
                         "SpMV A*dx");
        if (y.n)
            dual_trial<<<dy.n, 256, 0, stream.value>>>(y.n, y.p, trial_y.p, ax.p, previous_ax.p,
                                                       rl.p, ru.p, sigma, state.p, dy.p, cross.p);
        check(cudaGetLastError(), "dual trial kernel");
        decide<<<1, 256, 0, stream.value>>>(dx.n, dy.n, dx.p, dy.p, cross.p, tau, sigma, adaptive,
                                            halpern, residual_restarts, state.p);
        check(cudaGetLastError(), "step decision kernel");
        if (x.n)
            commit_primal<<<dx.n, 256, 0, stream.value>>>(x.n, x.p, trial_x.p, xa.p, state.p,
                                                          anchor_x.p, halpern, reflection);
        if (y.n)
            commit_dual<<<dy.n, 256, 0, stream.value>>>(y.n, y.p, trial_y.p, ya.p, previous_ax.p,
                                                        ax.p, state.p, anchor_y.p, anchor_ax.p,
                                                        halpern, reflection);
        check(cudaGetLastError(), "commit kernels");
    }
    int advance(int count, double tau, double sigma) override {
        if (quadratic) {
            double safety = std::sqrt(tau * sigma) * a_norm + tau * q_norm;
            double scale = safety > 0 ? std::min(1., .9 / safety) : 1.;
            tau *= scale;
            sigma *= scale;
        }
        if (graphs && (!graph.executable || tau != graph_tau || sigma != graph_sigma)) {
            graph.clear();
            // Warm sparse-library lazy initialization without modifying iteration state.
            double one = 1, zero = 0;
            if (a.nnz)
                sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, at.d,
                                          vy.d, &zero, vaty.d, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                          workspace->p),
                             "graph warmup");
            check(cudaDeviceSynchronize(), "graph warmup sync");
            check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeGlobal),
                  "graph begin capture");
            try {
                launch_iteration(tau, sigma);
            } catch (...) {
                cudaGraph_t failed = nullptr;
                cudaStreamEndCapture(stream.value, &failed);
                if (failed)
                    cudaGraphDestroy(failed);
                throw;
            }
            check(cudaStreamEndCapture(stream.value, &graph.graph), "graph end capture");
            check(cudaGraphInstantiate(&graph.executable, graph.graph, 0), "graph instantiate");
            graph_tau = tau;
            graph_sigma = sigma;
        }
        for (int i = 0; i < count; i++) {
            if (graphs)
                check(cudaGraphLaunch(graph.executable, stream.value), "PDHG graph launch");
            else
                launch_iteration(tau, sigma);
        }
        check(cudaStreamSynchronize(stream.value), "iteration chunk sync");
        std::vector<double> counters;
        state.download(counters); // Synchronizes this monitoring chunk, not every trial.
        int progress = int(int64_t(counters[2]) - accepted);
        accepted = int64_t(counters[2]);
        rejected = int64_t(counters[3]);
        request_restart = counters[9] != 0;
        empty_chunks = progress ? 0 : empty_chunks + 1;
        if (empty_chunks >= 4 || (!adaptive && !halpern && progress != count))
            throw std::runtime_error("CUDA PDHG backtracking stalled or produced nonfinite steps");
        return progress;
    }
    void candidates(std::vector<double> &px, std::vector<double> &py, std::vector<double> &pax,
                    std::vector<double> &pay) override {
        check(cudaStreamSynchronize(stream.value), "candidate sync");
        x.download(px);
        y.download(py);
        xa.download(pax);
        ya.download(pay);
        // Match the CPU activity rebasing at candidate checkpoints.
        refresh_activity(px);
    }
    void refresh_activity(const std::vector<double> &px) {
        xb.upload(px);
        double one = 1, zero = 0;
        if (a.nnz) {
            sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a.d, vx.d,
                                      &zero, vax.d, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                      workspace->p),
                         "refresh activity");
            check(cudaStreamSynchronize(stream.value), "activity sync");
            check(cudaMemcpy(previous_ax.p, ax.p, ax.n * sizeof(double), cudaMemcpyDeviceToDevice),
                  "refresh activity copy");
        }
    }
    void reset(const std::vector<double> &px, const std::vector<double> &py) override {
        x.upload(px);
        xb.upload(px);
        y.upload(py);
        xa.zero();
        ya.zero();
        refresh_activity(px);
        if (halpern) {
            anchor_x.upload(px);
            anchor_y.upload(py);
            check(cudaMemcpy(anchor_ax.p, previous_ax.p, previous_ax.n * sizeof(double),
                             cudaMemcpyDeviceToDevice),
                  "anchor activity");
            check(cudaMemset(state.p + 6, 0, 5 * sizeof(double)), "reset Halpern epoch");
            request_restart = false;
        }
        check(cudaMemset(state.p + 1, 0, sizeof(double)), "reset averaging mass");
        check(cudaDeviceSynchronize(), "backend reset sync");
    }
};
} // namespace
bool cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
std::string cuda_description() {
    if (!cuda_available())
        return "No CUDA device";
    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, 0), "GPU properties");
    return std::string(p.name) + " (" + std::to_string(p.totalGlobalMem / (1024 * 1024)) + " MiB)";
}
std::unique_ptr<IterationBackend>
cuda_backend(const Model &m, const std::vector<double> &x, const std::vector<double> &y,
             bool adaptive, bool halpern, double reflection, bool residual_restarts, bool graphs,
             const std::string &indices, const std::string &precision) {
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
    long double estimate =
        48.L * (m.A.value.size() + m.Q.value.size()) + 240.L * (m.A.rows + m.A.cols + 2);
    if (estimate > .8L * free)
        throw std::runtime_error(
            "Estimated GPU storage exceeds 80% of available memory; use --device cpu");
    if (halpern && adaptive)
        throw std::runtime_error("Halpern requires a fixed PDHG operator");
    return std::make_unique<CudaBackend>(m, x, y, adaptive, halpern, reflection, residual_restarts,
                                         graphs, indices, precision);
}
} // namespace vantage
