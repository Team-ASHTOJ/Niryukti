#include "internal.hpp"
namespace vantage {
class CpuBackend final : public IterationBackend {
    const Model &m;
    Sparse at;
    std::vector<double> x, y, xbar, xa, ya, aty;
    int64_t samples = 0;

  public:
    CpuBackend(const Model &m_, const std::vector<double> &x_, const std::vector<double> &y_)
        : m(m_), at(m.A.transpose()), x(x_), y(y_), xbar(x_), xa(x.size()), ya(y.size()),
          aty(x.size()) {}
    void advance(int count, double tau, double sigma) override {
        for (int it = 0; it < count; it++) {
#ifdef _OPENMP
#pragma omp parallel for if (m.A.cols > 10000)
#endif
            for (int64_t j = 0; j < m.A.cols; j++) {
                double g = m.c[j];
                for (auto k = at.ptr[j]; k < at.ptr[j + 1]; k++)
                    g += at.value[k] * y[at.index[k]];
                double next = std::clamp((x[j] - tau * g) / (1 + tau * m.q[j]), m.lb[j], m.ub[j]);
                xbar[j] = 2 * next - x[j];
                x[j] = next;
            }
#ifdef _OPENMP
#pragma omp parallel for if (m.A.rows > 10000)
#endif
            for (int64_t i = 0; i < m.A.rows; i++) {
                double ax = 0;
                for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++)
                    ax += m.A.value[k] * xbar[m.A.index[k]];
                double z = y[i] + sigma * ax;
                y[i] = std::max(0., z - sigma * m.ru[i]) + std::min(0., z - sigma * m.rl[i]);
            }
            samples++;
            double w = 1. / samples;
#ifdef _OPENMP
#pragma omp parallel for if (m.A.cols > 10000)
#endif
            for (int64_t j = 0; j < m.A.cols; j++)
                xa[j] += (x[j] - xa[j]) * w;
#ifdef _OPENMP
#pragma omp parallel for if (m.A.rows > 10000)
#endif
            for (int64_t i = 0; i < m.A.rows; i++)
                ya[i] += (y[i] - ya[i]) * w;
        }
    }
    void candidates(std::vector<double> &a, std::vector<double> &b, std::vector<double> &c,
                    std::vector<double> &d) override {
        a = x;
        b = y;
        c = xa;
        d = ya;
    }
    void reset(const std::vector<double> &a, const std::vector<double> &b) override {
        x = a;
        y = b;
        xbar = a;
        std::fill(xa.begin(), xa.end(), 0);
        std::fill(ya.begin(), ya.end(), 0);
        samples = 0;
    }
};
std::unique_ptr<IterationBackend> cpu_backend(const Model &m, const std::vector<double> &x,
                                              const std::vector<double> &y) {
    return std::make_unique<CpuBackend>(m, x, y);
}
} // namespace vantage
