#include "internal.hpp"
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace vantage {
volatile std::sig_atomic_t interrupted = 0;
Result solve(const Model &m, const Options &o) {
    auto overall_start = Clock::now();
    m.validate();
    if (!(o.tol > 0 && o.tol < 1) || !std::isfinite(o.tol) || o.iteration_limit < 0 ||
        o.node_limit < 0 || o.time_limit < 0 || std::isnan(o.time_limit) || o.check_every < 1 ||
        o.scaling_passes < 0 || o.scaling_passes > 20 || o.threads < 1 || o.mip_gap < 0 ||
        !std::isfinite(o.mip_gap) || o.integer_tol <= 0 || o.integer_tol >= .5)
        throw std::runtime_error("Invalid solver options");
    if (o.device != "auto" && o.device != "cpu" && o.device != "cuda")
        throw std::runtime_error("Unknown device");
    if (!o.initial_x.empty() && o.initial_x.size() != m.c.size())
        throw std::runtime_error("Warm-start primal dimension mismatch");
    if (!o.initial_y.empty() && o.initial_y.size() != m.rl.size())
        throw std::runtime_error("Warm-start dual dimension mismatch");
    for (auto v : o.initial_x)
        if (!std::isfinite(v))
            throw std::runtime_error("Nonfinite warm start");
    for (auto v : o.initial_y)
        if (!std::isfinite(v))
            throw std::runtime_error("Nonfinite warm start");
#ifdef _OPENMP
    omp_set_num_threads(o.threads);
#endif
    if (std::any_of(m.q.begin(), m.q.end(), [](double q) { return q < 0; })) {
        Result r;
        r.status = "UNSUPPORTED";
        r.message = "Nonconvex quadratic objective";
        return r;
    }
    if (m.is_mip() && m.is_qp()) {
        Result r;
        r.status = "UNSUPPORTED";
        r.message = "MIQP is not implemented";
        return r;
    }
    if (o.device == "cuda" && !cuda_available()) {
        Result r;
        r.status = "UNSUPPORTED";
        r.message = cuda_description();
        return r;
    }
    auto result = m.is_mip() ? solve_mip(m, o) : solve_continuous(m, o);
    result.seconds = elapsed(overall_start);
    return result;
}
Result solve_continuous(const Model &original, const Options &o) {
    auto start = Clock::now();
    Result r;
    Verifier verifier(original);
    r.status = "ITERATION_LIMIT";
    auto prep = prepare(original, o);
    r.preprocess_seconds = elapsed(start);
    if (!prep.failure.empty()) {
        r.status = prep.failure;
        r.message = prep.reason;
        r.seconds = elapsed(start);
        return r;
    }
    auto &m = prep.model;
    r.removed_columns = original.c.size() - m.c.size();
    r.removed_rows = original.rl.size() - m.rl.size();
    std::vector<double> x(m.c.size()), y(m.rl.size());
    for (size_t j = 0; j < x.size(); j++)
        x[j] =
            std::clamp(o.initial_x.empty() ? 0 : o.initial_x[prep.cols[j]] / prep.column_scale[j],
                       m.lb[j], m.ub[j]);
    for (size_t i = 0; i < y.size(); i++)
        if (!o.initial_y.empty())
            y[i] = o.initial_y[prep.rows[i]] / (prep.row_scale[i] * prep.objective_scale);
    r.x = prep.restore_x(x);
    r.y = prep.restore_y(y, original.rl.size());
    r.accuracy = verifier.evaluate(r.x, r.y);
    // Direct recession certificate for isolated linear columns, plus a checked feasible point.
    if (r.accuracy.finite && r.accuracy.primal_absolute == 0) {
        std::vector<bool> used(m.c.size(), false);
        for (auto j : m.A.index)
            used[j] = true;
        for (size_t j = 0; j < x.size(); j++)
            if (!used[j] && m.q[j] == 0 &&
                ((m.c[j] < 0 && m.ub[j] == inf) || (m.c[j] > 0 && m.lb[j] == -inf))) {
                r.status = "UNBOUNDED";
                r.message =
                    "Feasible point and isolated improving recession direction: " + m.names[j];
                r.seconds = elapsed(start);
                return r;
            }
    }
    if (r.accuracy.finite && r.accuracy.kkt <= o.tol) {
        r.status = "OPTIMAL";
        r.seconds = elapsed(start);
        return r;
    }
    if (o.time_limit == 0) {
        r.status = "TIME_LIMIT";
        r.seconds = elapsed(start);
        return r;
    }
    // ||A||_2 <= sqrt(||A||_1 ||A||_infinity): conservative guaranteed step product.
    std::vector<double> colsum(x.size());
    double rowmax = 0;
    for (size_t i = 0; i < y.size(); i++) {
        double s = 0;
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++) {
            double v = std::abs(m.A.value[k]);
            s += v;
            colsum[m.A.index[k]] += v;
        }
        rowmax = std::max(rowmax, s);
    }
    double colmax = 0;
    for (auto s : colsum)
        colmax = std::max(colmax, s);
    double norm = std::sqrt(rowmax) * std::sqrt(colmax);
    double step = norm > 0 ? .9 / norm : 1, weight = 1;
    if (o.adaptive) {
        long double c2 = 0, b2 = 0;
        for (auto c : m.c)
            c2 += (long double)c * c;
        for (size_t i = 0; i < m.rl.size(); i++) {
            double b = std::clamp(0., m.rl[i], m.ru[i]);
            b2 += (long double)b * b;
        }
        if (c2 > 1e-24 && b2 > 1e-24)
            weight = std::clamp(std::sqrt(double(c2 / b2)), 1e-4, 1e4);
    }
    bool usecuda = o.device == "cuda" ||
                   (o.device == "auto" && cuda_available() && m.A.value.size() >= 100000);
    r.backend = usecuda ? "cuda" : "cpu";
    r.device_name = usecuda ? cuda_description() : "CPU";
    auto transfer = Clock::now();
    auto backend = usecuda ? cuda_backend(m, x, y, o.adaptive) : cpu_backend(m, x, y, o.adaptive);
    r.transfer_seconds = elapsed(transfer);
    double restart_kkt = r.accuracy.kkt;
    int64_t since_restart = 0;
    std::vector<double> epochx = x, epochy = y;
    while (r.iterations < o.iteration_limit) {
        if (interrupted) {
            r.status = "INTERRUPTED";
            break;
        }
        if (elapsed(start) >= o.time_limit) {
            r.status = "TIME_LIMIT";
            break;
        }
        int count = int(std::min<int64_t>(o.check_every, o.iteration_limit - r.iterations));
        auto tick = Clock::now();
        int accepted = backend->advance(count, step / weight, step * weight);
        r.iteration_seconds += elapsed(tick);
        r.iterations += accepted;
        since_restart += accepted;
        r.rejected_steps = backend->rejected_steps();
        tick = Clock::now();
        std::vector<double> cx, cy, ax, ay;
        backend->candidates(cx, cy, ax, ay);
        auto ox = prep.restore_x(cx), oy = prep.restore_y(cy, original.rl.size());
        auto ca = verifier.evaluate(ox, oy, false);
        auto aox = prep.restore_x(ax), aoy = prep.restore_y(ay, original.rl.size());
        auto aa = verifier.evaluate(aox, aoy, false);
        bool avg = aa.kkt < ca.kkt;
        if (avg) {
            cx.swap(ax);
            cy.swap(ay);
            ox.swap(aox);
            oy.swap(aoy);
            ca = aa;
        }
        if (ca.finite && ca.kkt < r.accuracy.kkt) {
            r.x = ox;
            r.y = oy;
            r.accuracy = ca;
        }
        r.verification_seconds += elapsed(tick);
        if (o.verbose)
            std::cerr << "iter=" << r.iterations << " objective=" << original.sense * ca.objective
                      << " primal=" << ca.primal << " dual=" << ca.dual << " gap=" << ca.gap
                      << " seconds=" << elapsed(start) << '\n';
        if (!ca.finite) {
            r.status = "NUMERICAL_ERROR";
            r.message = "Nonfinite iterate or diagnostic";
            break;
        }
        if (ca.kkt <= o.tol)
            ca = verifier.evaluate(ox, oy);
        if (ca.kkt <= o.tol) {
            r.x = std::move(ox);
            r.y = std::move(oy);
            r.accuracy = ca;
            r.status = "OPTIMAL";
            break;
        }
        // A positive lower bound for the zero-objective feasibility problem is a
        // contradiction: every feasible point has objective zero. Verify in original units.
        if (r.iterations >= 1000 && ca.primal > o.tol) {
            auto ray = oy;
            double norm_y = 0;
            for (double v : ray)
                norm_y = std::max(norm_y, std::abs(v));
            if (norm_y > 0) {
                for (double &v : ray)
                    v /= norm_y;
                double margin = verifier.infeasibility_bound(ray);
                if (margin > 0 && std::isfinite(margin)) {
                    r.status = "INFEASIBLE";
                    r.message = "Independently verified box/row Farkas certificate";
                    r.infeasibility_ray = std::move(ray);
                    r.certificate_margin = margin;
                    break;
                }
            }
        }
        if (o.restart && since_restart >= o.check_every * 2 &&
            (ca.kkt < .5 * restart_kkt ||
             since_restart >= std::max<int64_t>(2000, r.iterations / 2))) {
            if (o.adaptive) {
                long double dx = 0, dy = 0;
                for (size_t j = 0; j < cx.size(); j++)
                    dx += (long double)(cx[j] - epochx[j]) * (cx[j] - epochx[j]);
                for (size_t i = 0; i < cy.size(); i++)
                    dy += (long double)(cy[i] - epochy[i]) * (cy[i] - epochy[i]);
                if (dx > 1e-24 && dy > 1e-24) {
                    double target = std::clamp(std::sqrt(double(dy / dx)), 1e-4, 1e4);
                    weight = std::sqrt(weight * target);
                }
            }
            backend->reset(cx, cy);
            epochx = cx;
            epochy = cy;
            restart_kkt = ca.kkt;
            since_restart = 0;
            r.restarts++;
        }
    }
    auto check = Clock::now();
    r.accuracy = verifier.evaluate(r.x, r.y);
    r.verification_seconds += elapsed(check);
    if (r.status == "OPTIMAL" && (!r.accuracy.finite || r.accuracy.kkt > o.tol))
        r.status = "NUMERICAL_ERROR";
    r.seconds = elapsed(start);
    return r;
}
} // namespace vantage
