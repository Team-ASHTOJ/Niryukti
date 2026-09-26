#include "internal.hpp"
#include <stdexcept>
namespace vantage {
class CpuBackend final : public IterationBackend {
    const Model &m;
    Sparse at;
    std::vector<double> x, y, delta_x, xa, ya, gradient, trial_x, trial_y, ax, next_ax;
    bool adaptive, halpern;
    double a_norm = 0, q_norm = 0;
    std::vector<double> anchor_x, anchor_y, anchor_ax;
    int64_t epoch_steps = 0;
    double reflection;
    bool fixed_point_restarts, request_restart = false;
    double initial_residual = inf, previous_residual = inf;
    double factor = 1, average_mass = 0;
    int64_t accepted = 0, rejected = 0;

  public:
    CpuBackend(const Model &model, const std::vector<double> &px, const std::vector<double> &py,
               bool adapt, bool anchored, double reflect, bool residual_restarts)
        : m(model), at(m.A.transpose()), x(px), y(py), delta_x(px), xa(px.size()), ya(py.size()),
          gradient(px.size()), trial_x(px.size()), trial_y(py.size()), ax(m.A.multiply(px)),
          next_ax(py.size()), adaptive(adapt), halpern(anchored), reflection(reflect),
          fixed_point_restarts(residual_restarts) {
        if (!m.Q.value.empty()) {
            adaptive = false;
            q_norm = m.quadratic_norm_bound();
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
        if (halpern) {
            anchor_x = x;
            anchor_y = y;
            anchor_ax = ax;
        }
    }
    int64_t rejected_steps() const override {
        return rejected;
    }
    bool restart_requested() const override {
        return request_restart;
    }
    int advance(int count, double primal_step, double dual_step) override {
        if (!m.Q.value.empty()) {
            double safety = std::sqrt(primal_step * dual_step) * a_norm + primal_step * q_norm;
            double scale = safety > 0 ? std::min(1., .9 / safety) : 1.;
            primal_step *= scale;
            dual_step *= scale;
        }
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
            if (!m.Q.value.empty()) {
                auto qx = m.Q.multiply(x);
                for (size_t j = 0; j < x.size(); ++j)
                    gradient[j] += qx[j] + m.q[j] * x[j];
            }
            bool success = false;
            for (int attempt = 0; attempt < 40; attempt++) {
                double tau = primal_step * factor, sigma = dual_step * factor;
                double dx2 = 0, dy2 = 0, coupling = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dx2) if (m.A.cols > 10000)
#endif
                for (int64_t j = 0; j < m.A.cols; j++) {
                    trial_x[j] = std::clamp((x[j] - tau * gradient[j]) /
                                                (1 + tau * (m.Q.value.empty() ? m.q[j] : 0)),
                                            m.lb[j], m.ub[j]);
                    double dx = trial_x[j] - x[j];
                    dx2 += dx * dx;
                    delta_x[j] = dx; // Multiply the displacement, avoiding cancellation in A*dx.
                }
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dy2, coupling) if (m.A.rows > 10000)
#endif
                for (int64_t i = 0; i < m.A.rows; i++) {
                    double delta = 0;
                    for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++)
                        delta += m.A.value[k] * delta_x[m.A.index[k]];
                    double bar = ax[i] + 2 * delta;
                    double z = y[i] + sigma * bar;
                    trial_y[i] =
                        std::max(0., z - sigma * m.ru[i]) + std::min(0., z - sigma * m.rl[i]);
                    double dy = trial_y[i] - y[i];
                    dy2 += dy * dy;
                    if (!std::isfinite(bar) || !std::isfinite(trial_y[i]))
                        dy2 = inf;
                    coupling += dy * delta;
                    next_ax[i] = ax[i] + delta;
                }
                double energy = dx2 / tau + dy2 / sigma;
                double limit = std::abs(coupling) > 0 ? energy / (2 * std::abs(coupling)) : inf;
                bool finite = std::isfinite(energy) && std::isfinite(coupling);
                double used_factor = factor;
                bool accept = finite && (!adaptive || limit >= 1);
                if (adaptive) {
                    double k = double(accepted + 2);
                    double multiplier =
                        std::min((1 - std::pow(k, -.3)) * limit, 1 + std::pow(k, -.6));
                    if (!finite || !std::isfinite(multiplier) || multiplier <= 0)
                        multiplier = .5;
                    factor = std::clamp(factor * multiplier, 1e-12, 1e12);
                }
                if (!accept) {
                    rejected++;
                    if (!adaptive)
                        throw std::runtime_error("Nonfinite PDHG step");
                    continue;
                }
                x.swap(trial_x);
                y.swap(trial_y);
                ax.swap(next_ax);
                accepted++;
                if (halpern) {
                    // Preserve T(z) as the second stopping/restart candidate.
                    xa = x;
                    ya = y;
                    double lambda = 1 / (double(epoch_steps) + 2);
                    for (size_t j = 0; j < x.size(); j++)
                        x[j] = lambda * anchor_x[j] +
                               (1 - lambda) * ((1 + reflection) * x[j] - reflection * trial_x[j]);
                    for (size_t i = 0; i < y.size(); i++) {
                        y[i] = lambda * anchor_y[i] +
                               (1 - lambda) * ((1 + reflection) * y[i] - reflection * trial_y[i]);
                        ax[i] = lambda * anchor_ax[i] +
                                (1 - lambda) * ((1 + reflection) * ax[i] - reflection * next_ax[i]);
                    }
                    double residual = std::sqrt(std::max(0., energy - 2 * coupling));
                    if (epoch_steps == 0)
                        initial_residual = residual;
                    epoch_steps++;
                    request_restart =
                        fixed_point_restarts && epoch_steps >= 10 &&
                        (residual <= .2 * initial_residual ||
                         (residual <= .8 * initial_residual && residual > previous_residual) ||
                         epoch_steps >= std::max<int64_t>(10, accepted / 2));
                    previous_residual = residual;
                    if (request_restart)
                        return it + 1;
                    success = true;
                    break;
                }
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
        // Rebase the activity recurrence at monitoring checkpoints without resetting averages.
        ax = m.A.multiply(x);
    }
    void reset(const std::vector<double> &px, const std::vector<double> &py) override {
        x = px;
        y = py;
        ax = m.A.multiply(x);
        if (halpern) {
            anchor_x = x;
            anchor_y = y;
            anchor_ax = ax;
            epoch_steps = 0;
            initial_residual = previous_residual = inf;
            request_restart = false;
        }
        average_mass = 0;
        std::fill(xa.begin(), xa.end(), 0);
        std::fill(ya.begin(), ya.end(), 0);
    }
};
std::unique_ptr<IterationBackend> cpu_backend(const Model &m, const std::vector<double> &x,
                                              const std::vector<double> &y, bool adaptive,
                                              bool halpern, double reflection,
                                              bool fixed_point_restarts) {
    if (halpern && adaptive)
        throw std::runtime_error("Halpern requires a fixed PDHG operator");
    if (!std::isfinite(reflection) || reflection < 0 || reflection > 1)
        throw std::runtime_error("Invalid Halpern reflection");
    return std::make_unique<CpuBackend>(m, x, y, adaptive, halpern, reflection,
                                        fixed_point_restarts);
}
} // namespace vantage
