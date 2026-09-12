#include "internal.hpp"
#include <iostream>
#include <stdexcept>
namespace vantage {
class CpuBackend final : public IterationBackend {
    const Model &m;
    Sparse at;
    std::vector<double> x, y, xbar, xa, ya, gradient, trial_x, trial_y, ax, next_ax;
    bool adaptive;
    double factor = 1, average_mass = 0;
    int64_t accepted = 0, rejected = 0;

  public:
    CpuBackend(const Model &model, const std::vector<double> &px, const std::vector<double> &py,
               bool adapt)
        : m(model), at(m.A.transpose()), x(px), y(py), xbar(px), xa(px.size()), ya(py.size()),
          gradient(px.size()), trial_x(px.size()), trial_y(py.size()), ax(m.A.multiply(px)),
          next_ax(py.size()), adaptive(adapt) {}
    int64_t rejected_steps() const override {
        return rejected;
    }
    int advance(int count, double primal_step, double dual_step) override {
        for (int it = 0; it < count; it++) {
#ifdef _OPENMP
#pragma omp parallel for if (m.A.cols > 10000)
#endif
            for (int64_t j = 0; j < m.A.cols; j++) {
                double g = m.c[j];
                for (auto k = at.ptr[j]; k < at.ptr[j + 1]; k++)
                    g += at.value[k] * y[at.index[k]];
                gradient[j] = g;
            }
            bool success = false;
            for (int attempt = 0; attempt < 40; attempt++) {
                double tau = primal_step * factor, sigma = dual_step * factor;
                double dx2 = 0, dy2 = 0, coupling = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dx2) if (m.A.cols > 10000)
#endif
                for (int64_t j = 0; j < m.A.cols; j++) {
                    trial_x[j] = std::clamp((x[j] - tau * gradient[j]) / (1 + tau * m.q[j]),
                                            m.lb[j], m.ub[j]);
                    double dx = trial_x[j] - x[j];
                    dx2 += dx * dx;
                    xbar[j] = 2 * trial_x[j] - x[j];
                }
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dy2, coupling) if (m.A.rows > 10000)
#endif
                for (int64_t i = 0; i < m.A.rows; i++) {
                    double bar = 0;
                    for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++)
                        bar += m.A.value[k] * xbar[m.A.index[k]];
                    double z = y[i] + sigma * bar;
                    trial_y[i] =
                        std::max(0., z - sigma * m.ru[i]) + std::min(0., z - sigma * m.rl[i]);
                    double dy = trial_y[i] - y[i];
                    dy2 += dy * dy;
                    coupling += dy * (bar - ax[i]) * .5;
                    next_ax[i] = (bar + ax[i]) * .5;
                }
                double energy = dx2 / tau + dy2 / sigma;
                double limit = std::abs(coupling) > 0 ? energy / (2 * std::abs(coupling)) : inf;
                bool finite = std::isfinite(energy) && std::isfinite(coupling);
                double used_factor = factor;
                bool accept = finite && (!adaptive || limit >= 0.5);
                if (adaptive) {
                    double k = double(accepted + 2);
                    double multiplier =
                        std::min((1 - std::pow(k, -.3)) * limit, 1 + std::pow(k, -.6));
                    if (!finite || !std::isfinite(multiplier) || multiplier <= 0)
                        multiplier = .5;
                    factor = std::clamp(factor * multiplier, 1e-12, 1e12);
                }
                if (!accept) {
                    if (attempt < 5) {
                        std::cerr << "PDHG reject: attempt=" << attempt
                                  << " factor=" << factor
                                  << " tau=" << tau
                                  << " sigma=" << sigma
                                  << " energy=" << energy
                                  << " coupling=" << coupling
                                  << " limit=" << limit
                                  << " finite=" << finite
                                  << "\\n";
                    }
                    rejected++;
                    if (!adaptive)
                        throw std::runtime_error("Nonfinite PDHG step");
                    continue;
                }
                x.swap(trial_x);
                y.swap(trial_y);
                ax.swap(next_ax);
                accepted++;
                average_mass += used_factor;
                double w = used_factor / average_mass;
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
                success = true;
                break;
            }
            if (!success)
                throw std::runtime_error("PDHG backtracking exhausted");
        }
        return count;
    }
    void candidates(std::vector<double> &cx, std::vector<double> &cy, std::vector<double> &avgx,
                    std::vector<double> &avgy) override {
        cx = x;
        cy = y;
        avgx = xa;
        avgy = ya;
    }
    void reset(const std::vector<double> &px, const std::vector<double> &py) override {
        x = px;
        y = py;
        ax = m.A.multiply(x);
        average_mass = 0;
        std::fill(xa.begin(), xa.end(), 0);
        std::fill(ya.begin(), ya.end(), 0);
    }
};
std::unique_ptr<IterationBackend> cpu_backend(const Model &m, const std::vector<double> &x,
                                              const std::vector<double> &y, bool adaptive) {
    return std::make_unique<CpuBackend>(m, x, y, adaptive);
}
} // namespace vantage
