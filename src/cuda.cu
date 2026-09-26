#include "internal.hpp"
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
    Buffer<int64_t> ptr, index;
    Buffer<double> values;
    cusparseSpMatDescr_t d = nullptr;
    explicit Matrix(const Sparse &s) : ptr(s.ptr), index(s.index), values(s.value) {
        sparse_check(cusparseCreateCsr(&d, s.rows, s.cols, s.value.size(), ptr.p, index.p, values.p,
                                       CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                     "create CSR");
    }
    ~Matrix() {
        if (d)
            cusparseDestroySpMat(d);
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
                             const double *aty, const double *c, const double *q, const double *l,
                             const double *u, double base_tau, const double *state,
                             double *partial) {
    auto j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    double square = 0;
    if (j < n) {
        double tau = base_tau * state[0];
        double v = fmin(u[j], fmax(l[j], (x[j] - tau * (c[j] + aty[j])) / (1 + tau * q[j])));
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
                       double tau, double sigma, bool adaptive, double *state) {
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
        } else
            state[3] += 1;
    }
}
__global__ void commit_primal(int64_t n, double *x, const double *trial, double *avg,
                              const double *state) {
    auto j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < n && state[4]) {
        x[j] = trial[j];
        avg[j] += (x[j] - avg[j]) * state[5];
    }
}
__global__ void commit_dual(int64_t n, double *y, const double *trial, double *avg,
                            double *previous_ax, const double *delta, const double *state) {
    auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && state[4]) {
        y[i] = trial[i];
        avg[i] += (y[i] - avg[i]) * state[5];
        previous_ax[i] += delta[i];
    }
}
class CudaBackend final : public IterationBackend {
    Handle handle;
    Matrix a, at;
    Buffer<double> x, y, xb, xa, ya, ax, aty, c, q, lb, ub, rl, ru;
    Buffer<double> trial_x, trial_y, previous_ax, dx, dy, cross, state;
    bool adaptive;
    int64_t accepted = 0, rejected = 0;
    int empty_chunks = 0;
    Vector vx, vy, vax, vaty;
    std::unique_ptr<Buffer<char>> workspace;

  public:
    CudaBackend(const Model &m, const std::vector<double> &px, const std::vector<double> &py,
                bool adapt)
        : a(m.A), at(m.A.transpose()), x(px), y(py), xb(px), xa(px.size()), ya(py.size()),
          ax(py.size()), aty(px.size()), c(m.c), q(m.q), lb(m.lb), ub(m.ub), rl(m.rl), ru(m.ru),
          trial_x(px.size()), trial_y(py.size()), previous_ax(m.A.multiply(px)),
          dx((px.size() + 255) / 256), dy((py.size() + 255) / 256), cross((py.size() + 255) / 256),
          state(std::vector<double>{1, 0, 0, 0, 0, 0}), adaptive(adapt), vx(xb.n, xb.p),
          vy(y.n, y.p), vax(ax.n, ax.p), vaty(aty.n, aty.p) {
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
        workspace = std::make_unique<Buffer<char>>(std::max(s1, s2));
    }

    int64_t rejected_steps() const override {
        return rejected;
    }
    int advance(int count, double tau, double sigma) override {
        double one = 1, zero = 0;
        for (int i = 0; i < count; i++) {
            if (a.values.n)
                sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, at.d,
                                          vy.d, &zero, vaty.d, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                          workspace->p),
                             "SpMV At*y");
            if (x.n)
                primal_trial<<<dx.n, 256>>>(x.n, x.p, trial_x.p, xb.p, aty.p, c.p, q.p, lb.p, ub.p,
                                            tau, state.p, dx.p);
            check(cudaGetLastError(), "primal trial kernel");
            if (a.values.n)
                sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a.d,
                                          vx.d, &zero, vax.d, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                          workspace->p),
                             "SpMV A*dx");
            if (y.n)
                dual_trial<<<dy.n, 256>>>(y.n, y.p, trial_y.p, ax.p, previous_ax.p, rl.p, ru.p,
                                          sigma, state.p, dy.p, cross.p);
            check(cudaGetLastError(), "dual trial kernel");
            decide<<<1, 256>>>(dx.n, dy.n, dx.p, dy.p, cross.p, tau, sigma, adaptive, state.p);
            check(cudaGetLastError(), "step decision kernel");
            if (x.n)
                commit_primal<<<dx.n, 256>>>(x.n, x.p, trial_x.p, xa.p, state.p);
            if (y.n)
                commit_dual<<<dy.n, 256>>>(y.n, y.p, trial_y.p, ya.p, previous_ax.p, ax.p, state.p);
            check(cudaGetLastError(), "commit kernels");
        }
        std::vector<double> counters;
        state.download(counters); // Synchronizes this monitoring chunk, not every trial.
        int progress = int(int64_t(counters[2]) - accepted);
        accepted = int64_t(counters[2]);
        rejected = int64_t(counters[3]);
        empty_chunks = progress ? 0 : empty_chunks + 1;
        if (empty_chunks >= 4 || (!adaptive && progress != count))
            throw std::runtime_error("CUDA PDHG backtracking stalled or produced nonfinite steps");
        return progress;
    }
    void candidates(std::vector<double> &px, std::vector<double> &py, std::vector<double> &pax,
                    std::vector<double> &pay) override {
        x.download(px);
        y.download(py);
        xa.download(pax);
        ya.download(pay);
    }
    void reset(const std::vector<double> &px, const std::vector<double> &py) override {
        x.upload(px);
        xb.upload(px);
        y.upload(py);
        xa.zero();
        ya.zero();
        // Recompute A*x at restarts to avoid accumulated recurrence roundoff.
        double one = 1, zero = 0;
        if (a.values.n) {
            sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a.d, vx.d,
                                      &zero, vax.d, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                      workspace->p),
                         "restart A*x");
            check(cudaMemcpy(previous_ax.p, ax.p, ax.n * sizeof(double), cudaMemcpyDeviceToDevice),
                  "restart activity");
        }
        check(cudaMemset(state.p + 1, 0, sizeof(double)), "reset averaging mass");
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
std::unique_ptr<IterationBackend> cuda_backend(const Model &m, const std::vector<double> &x,
                                               const std::vector<double> &y, bool adaptive) {
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
    long double estimate = 48.L * m.A.value.size() + 192.L * (m.A.rows + m.A.cols + 2);
    if (estimate > .8L * free)
        throw std::runtime_error(
            "Estimated GPU storage exceeds 80% of available memory; use --device cpu");
    return std::make_unique<CudaBackend>(m, x, y, adaptive);
}
} // namespace vantage
