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
__global__ void primal(int64_t n, double *x, double *xb, const double *aty, const double *c,
                       const double *q, const double *l, const double *u, double tau, double *avg,
                       double w) {
    auto j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < n) {
        double v = fmin(u[j], fmax(l[j], (x[j] - tau * (c[j] + aty[j])) / (1 + tau * q[j])));
        xb[j] = 2 * v - x[j];
        x[j] = v;
        avg[j] += (v - avg[j]) * w;
    }
}
__global__ void dual(int64_t n, double *y, const double *ax, const double *l, const double *u,
                     double sigma, double *avg, double w) {
    auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        double z = y[i] + sigma * ax[i];
        double v = fmax(0., z - sigma * u[i]) + fmin(0., z - sigma * l[i]);
        y[i] = v;
        avg[i] += (v - avg[i]) * w;
    }
}
class CudaBackend final : public IterationBackend {
    Handle handle;
    Matrix a, at;
    Buffer<double> x, y, xb, xa, ya, ax, aty, c, q, lb, ub, rl, ru;
    Vector vx, vy, vax, vaty;
    std::unique_ptr<Buffer<char>> workspace;
    int64_t samples = 0;

  public:
    CudaBackend(const Model &m, const std::vector<double> &px, const std::vector<double> &py)
        : a(m.A), at(m.A.transpose()), x(px), y(py), xb(px), xa(px.size()), ya(py.size()),
          ax(py.size()), aty(px.size()), c(m.c), q(m.q), lb(m.lb), ub(m.ub), rl(m.rl), ru(m.ru),
          vx(xb.n, xb.p), vy(y.n, y.p), vax(ax.n, ax.p), vaty(aty.n, aty.p) {
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
    void advance(int count, double tau, double sigma) override {
        double one = 1, zero = 0;
        for (int i = 0; i < count; i++) {
            double w = 1. / ++samples;
            if (a.values.n)
                sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, at.d,
                                          vy.d, &zero, vaty.d, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                          workspace->p),
                             "SpMV At*y");
            if (x.n)
                primal<<<(x.n + 255) / 256, 256>>>(x.n, x.p, xb.p, aty.p, c.p, q.p, lb.p, ub.p, tau,
                                                   xa.p, w);
            check(cudaGetLastError(), "primal kernel");
            if (a.values.n)
                sparse_check(cusparseSpMV(handle.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a.d,
                                          vx.d, &zero, vax.d, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                          workspace->p),
                             "SpMV A*x");
            if (y.n)
                dual<<<(y.n + 255) / 256, 256>>>(y.n, y.p, ax.p, rl.p, ru.p, sigma, ya.p, w);
            check(cudaGetLastError(), "dual kernel");
        }
        check(cudaDeviceSynchronize(), "PDHG chunk synchronize");
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
        samples = 0;
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
                                               const std::vector<double> &y) {
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
    long double estimate = 48.L * m.A.value.size() + 128.L * (m.A.rows + m.A.cols + 2);
    if (estimate > .8L * free)
        throw std::runtime_error(
            "Estimated GPU storage exceeds 80% of available memory; use --device cpu");
    return std::make_unique<CudaBackend>(m, x, y);
}
} // namespace vantage
