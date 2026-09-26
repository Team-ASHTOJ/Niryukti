#include "internal.hpp"
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace vantage {
namespace {
bool anchored_method(const Options &o) {
    return o.method == "halpern" || o.method == "rhpdhg" || o.method == "r2hpdhg";
}
} // namespace
volatile std::sig_atomic_t interrupted = 0;
Result solve(const Model &m, const Options &o) {
    auto overall_start = Clock::now();
    m.validate();
    if (!(o.tol > 0 && o.tol < 1) || !std::isfinite(o.tol) || o.iteration_limit < 0 ||
        o.node_limit < 0 || o.time_limit < 0 || std::isnan(o.time_limit) || o.check_every < 1 ||
        o.scaling_passes < 0 || o.scaling_passes > 20 || o.threads < 1 || o.mip_gap < 0 ||
        !std::isfinite(o.mip_gap) || !std::isfinite(o.integer_tol) || o.integer_tol <= 0 ||
        o.integer_tol >= .5 || o.power_iterations < 0 || o.power_iterations > 1000)
        throw std::runtime_error("Invalid solver options");
    if (o.device != "auto" && o.device != "cpu" && o.device != "cuda")
        throw std::runtime_error("Unknown device");
    if (o.method != "pdhg" && !anchored_method(o))
        throw std::runtime_error("Unknown method: use pdhg, halpern, rhpdhg or r2hpdhg");
    if (o.primal_weight != "displacement" && o.primal_weight != "pid")
        throw std::runtime_error("Unknown primal weight controller");
    if (o.primal_heuristic != "repair" && o.primal_heuristic != "pump" &&
        o.primal_heuristic != "rins" && o.primal_heuristic != "all")
        throw std::runtime_error("Unknown primal heuristic");
    if (o.polishing && m.is_qp())
        throw std::runtime_error("Feasibility polishing currently requires LP relaxations");
    if (o.scaling != "ruiz" && o.scaling != "combined")
        throw std::runtime_error("Unknown scaling: use ruiz or combined");
    if (o.branching != "fractional" && o.branching != "reliability")
        throw std::runtime_error("Unknown branching: use fractional or reliability");
    if (anchored_method(o) && (o.adaptive || m.is_qp() || m.is_mip()))
        throw std::runtime_error("Experimental Halpern supports continuous LP with --no-adaptive");
    if (o.gpu_indices != "auto" && o.gpu_indices != "32" && o.gpu_indices != "64")
        throw std::runtime_error("GPU indices must be auto, 32 or 64");
    if (o.matrix_precision != "fp64" && o.matrix_precision != "mixed")
        throw std::runtime_error("Matrix precision must be fp64 or mixed");
    if ((o.cuda_graphs || o.matrix_precision == "mixed") && o.device == "cpu")
        throw std::runtime_error("CUDA graphs and mixed matrix precision require a CUDA backend");
    if (anchored_method(o) && o.matrix_precision == "mixed")
        throw std::runtime_error("Fixed Halpern operators require FP64 matrices");
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
    if (m.Q.value.empty() && std::any_of(m.q.begin(), m.q.end(), [](double q) { return q < 0; })) {
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
    if (anchored_method(o) && (o.adaptive || original.is_qp() || original.is_mip()))
        throw std::runtime_error("Experimental Halpern supports continuous LP with --no-adaptive");
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
    if (o.power_iterations > 0 && m.Q.value.empty()) {
        r.operator_norm_estimate = power_norm(m.A, o.power_iterations);
        // A power estimate is not an upper bound. Only the backtracking path may
        // use it for larger initial steps; fixed-operator methods retain the bound.
        if (o.adaptive && r.operator_norm_estimate > 0)
            norm = std::min(norm, 1.05 * r.operator_norm_estimate);
    }
    double step = norm > 0 ? .9 / norm : 1, weight = 1;
    if ((o.adaptive || anchored_method(o)) && o.primal_weight != "pid") {
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
                   (o.device == "auto" && cuda_available() &&
                    (m.A.value.size() >= 100000 || o.cuda_graphs || o.matrix_precision == "mixed"));
    r.backend = usecuda ? "cuda" : "cpu";
    r.device_name = usecuda ? cuda_description() : "CPU";
    auto transfer = Clock::now();
    bool residual_restarts = o.method == "rhpdhg" || o.method == "r2hpdhg";
    auto backend = usecuda
                       ? cuda_backend(m, x, y, o.adaptive, anchored_method(o),
                                      o.method == "r2hpdhg" ? 1 : 0, residual_restarts && o.restart,
                                      o.cuda_graphs, o.gpu_indices, o.matrix_precision)
                       : cpu_backend(m, x, y, o.adaptive, anchored_method(o),
                                     o.method == "r2hpdhg" ? 1 : 0, residual_restarts && o.restart);
    r.transfer_seconds = elapsed(transfer);
    if (usecuda) {
        bool fits32 =
            m.A.rows <= INT32_MAX && m.A.cols <= INT32_MAX && m.A.value.size() <= INT32_MAX;
        r.gpu_index_bits = o.gpu_indices == "64" || !fits32 ? 64 : 32;
        r.graph_execution = o.cuda_graphs;
        r.matrix_precision = o.matrix_precision;
    }
    double restart_kkt = r.accuracy.kkt;
    int64_t since_restart = 0;
    std::vector<double> epochx = x, epochy = y;
    PrimalWeightController controller;
    int64_t next_polish = 100, previous_rejected = 0;
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
        auto rejected = backend->rejected_steps();
        r.rejected_steps += rejected - previous_rejected;
        previous_rejected = rejected;
        tick = Clock::now();
        std::vector<double> cx, cy, ax, ay;
        backend->candidates(cx, cy, ax, ay);
        std::vector<double> restart_x, restart_y;
        if (residual_restarts && backend->restart_requested()) {
            restart_x = ax;
            restart_y = ay;
        }
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
            // Candidate rays from iterates and epoch displacement. These are
            // heuristic extractions under adaptive/restarted iterations; only
            // the outward-rounded original-model bound certifies infeasibility.
            auto displacement = cy;
            for (size_t i = 0; i < displacement.size(); ++i)
                displacement[i] -= epochy[i];
            std::vector<std::vector<double>> rays;
            rays.push_back(oy);
            rays.push_back(aoy);
            rays.push_back(prep.restore_y(displacement, original.rl.size()));
            for (auto &ray : rays) {
                double norm_y = 0;
                bool finite = true;
                for (double v : ray) {
                    finite = finite && std::isfinite(v);
                    norm_y = std::max(norm_y, std::abs(v));
                }
                if (!finite || !(norm_y > 0))
                    continue;
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
            if (r.status == "INFEASIBLE")
                break;
        }
        if (o.polishing && !original.is_qp() && r.iterations >= next_polish) {
            next_polish = r.iterations <= std::numeric_limits<int64_t>::max() / 2
                              ? 2 * r.iterations
                              : std::numeric_limits<int64_t>::max();
            int64_t budget = std::min(r.iterations / 8, (o.iteration_limit - r.iterations) / 2);
            if (ca.gap <= .01 && budget > 0 && !interrupted && elapsed(start) < o.time_limit) {
                r.polishing_attempts++;
                Model primal = original;
                std::fill(primal.c.begin(), primal.c.end(), 0);
                primal.offset = 0;
                Options po = o;
                po.method = "pdhg";
                po.adaptive = true;
                po.polishing = false;
                po.initial_x = ox;
                po.initial_y.assign(original.rl.size(), 0);
                po.iteration_limit = budget;
                po.time_limit = std::max(0., o.time_limit - elapsed(start));
                auto accumulate = [&](const Result &aux) {
                    r.iterations += aux.iterations;
                    r.polishing_iterations += aux.iterations;
                    r.restarts += aux.restarts;
                    r.rejected_steps += aux.rejected_steps;
                    r.weight_updates += aux.weight_updates;
                    r.preprocess_seconds += aux.preprocess_seconds;
                    r.transfer_seconds += aux.transfer_seconds;
                    r.iteration_seconds += aux.iteration_seconds;
                    r.verification_seconds += aux.verification_seconds;
                };
                auto pr = solve_continuous(primal, po);
                accumulate(pr);
                if (pr.accuracy.finite && pr.accuracy.primal <= o.tol && !interrupted &&
                    elapsed(start) < o.time_limit) {
                    auto dual = dual_feasibility_model(original);
                    po.initial_x = oy;
                    po.initial_y.assign(original.c.size(), 0);
                    po.iteration_limit = std::min(budget, o.iteration_limit - r.iterations);
                    po.time_limit = std::max(0., o.time_limit - elapsed(start));
                    auto dr = solve_continuous(dual, po);
                    accumulate(dr);
                    if (dr.x.size() == original.rl.size()) {
                        auto polished = verifier.evaluate(pr.x, dr.x);
                        if (polished.finite && polished.kkt < r.accuracy.kkt) {
                            r.x = std::move(pr.x);
                            r.y = std::move(dr.x);
                            r.accuracy = polished;
                            if (polished.kkt <= o.tol) {
                                r.status = "OPTIMAL";
                                break;
                            }
                        }
                    }
                }
            }
        }
        bool restart_due = residual_restarts
                               ? backend->restart_requested()
                               : since_restart >= int64_t(o.check_every) * 2 &&
                                     (ca.kkt < .5 * restart_kkt ||
                                      since_restart >= std::max<int64_t>(2000, r.iterations / 2));
        if (o.restart && restart_due) {
            if (residual_restarts) {
                cx = std::move(restart_x);
                cy = std::move(restart_y);
            }
            if (o.adaptive || anchored_method(o) || o.primal_weight == "pid") {
                long double dx = 0, dy = 0;
                for (size_t j = 0; j < cx.size(); j++)
                    dx += (long double)(cx[j] - epochx[j]) * (cx[j] - epochx[j]);
                for (size_t i = 0; i < cy.size(); i++)
                    dy += (long double)(cy[i] - epochy[i]) * (cy[i] - epochy[i]);
                if (dx > 1e-24 && dy > 1e-24) {
                    if (o.primal_weight == "pid")
                        weight = controller.update(weight, dx, dy);
                    else {
                        double target = std::clamp(std::sqrt(double(dy / dx)), 1e-4, 1e4);
                        weight = std::sqrt(weight * target);
                    }
                    r.weight_updates++;
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
