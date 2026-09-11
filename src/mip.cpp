#include "internal.hpp"
#include <queue>
namespace vantage {
namespace {
struct Node {
    std::vector<double> lb, ub, x, y;
    double bound = -inf;
    int depth = 0;
};
// Conservative interval arithmetic for implied INTEGER bounds only. Row bounds stay
// unchanged, so continuous dual postsolve does not need new multiplier mappings.
long double outward(long double value, bool lower) {
    return std::nextafter(value, lower ? -std::numeric_limits<long double>::infinity()
                                       : std::numeric_limits<long double>::infinity());
}
bool propagate(const Model &m, Node &node, int64_t &tightened) {
    for (int pass = 0; pass < 5; pass++) {
        bool changed = false;
        for (int64_t row = 0; row < m.A.rows; row++) {
            auto begin = m.A.ptr[row], end = m.A.ptr[row + 1];
            size_t count = end - begin;
            std::vector<long double> low(count + 1), high(count + 1);
            for (size_t t = 0; t < count; t++) {
                auto k = begin + t;
                auto j = m.A.index[k];
                double a = m.A.value[k];
                auto lo = outward((long double)a * (a > 0 ? node.lb[j] : node.ub[j]), true);
                auto hi = outward((long double)a * (a > 0 ? node.ub[j] : node.lb[j]), false);
                low[t + 1] = outward(low[t] + lo, true);
                high[t + 1] = outward(high[t] + hi, false);
            }
            if (low[count] > m.ru[row] || high[count] < m.rl[row])
                return false;
            long double suffix_low = 0, suffix_high = 0;
            for (size_t t = count; t-- > 0;) {
                auto k = begin + t;
                auto j = m.A.index[k];
                double a = m.A.value[k];
                auto term_low = outward((long double)a * (a > 0 ? node.lb[j] : node.ub[j]), true);
                auto term_high = outward((long double)a * (a > 0 ? node.ub[j] : node.lb[j]), false);
                if (m.types[j] != VarType::Continuous) {
                    long double others_low = outward(low[t] + suffix_low, true);
                    long double others_high = outward(high[t] + suffix_high, false);
                    long double implied_low = -inf, implied_high = inf;
                    if (std::isfinite(m.rl[row]) && std::isfinite(others_high)) {
                        long double numerator = outward((long double)m.rl[row] - others_high, true);
                        if (a > 0)
                            implied_low = outward(numerator / a, true);
                        else
                            implied_high = outward(numerator / a, false);
                    }
                    if (std::isfinite(m.ru[row]) && std::isfinite(others_low)) {
                        long double numerator = outward((long double)m.ru[row] - others_low, false);
                        if (a > 0)
                            implied_high = outward(numerator / a, false);
                        else
                            implied_low = outward(numerator / a, true);
                    }
                    // Restrict tightening to exactly representable integer magnitudes.
                    if (std::abs(implied_low) < 0x1p52L) {
                        double bound = double(std::ceil(implied_low));
                        if (bound > node.lb[j]) {
                            node.lb[j] = bound;
                            changed = true;
                            tightened++;
                        }
                    }
                    if (std::abs(implied_high) < 0x1p52L) {
                        double bound = double(std::floor(implied_high));
                        if (bound < node.ub[j]) {
                            node.ub[j] = bound;
                            changed = true;
                            tightened++;
                        }
                    }
                    if (node.lb[j] > node.ub[j])
                        return false;
                }
                // The old interval remains a safe superset after any tightening above.
                suffix_low = outward(suffix_low + term_low, true);
                suffix_high = outward(suffix_high + term_high, false);
            }
        }
        if (!changed)
            break;
    }
    return true;
}

struct Compare {
    bool operator()(const Node &a, const Node &b) const {
        return a.bound > b.bound;
    }
};
} // namespace
Result solve_mip(const Model &original, const Options &options) {
    auto start = Clock::now();
    Result out;
    out.status = "NODE_LIMIT";
    Model relaxation = original;
    std::fill(relaxation.types.begin(), relaxation.types.end(), VarType::Continuous);
    Node root;
    root.lb = original.lb;
    root.ub = original.ub;
    root.x = options.initial_x;
    root.y = options.initial_y;
    for (size_t j = 0; j < original.c.size(); j++)
        if (original.types[j] != VarType::Continuous) {
            root.lb[j] = std::ceil(root.lb[j]);
            root.ub[j] = std::floor(root.ub[j]);
            if (root.lb[j] > root.ub[j]) {
                out.status = "INFEASIBLE";
                out.message = "Integer variable has empty integer domain";
                out.seconds = elapsed(start);
                return out;
            }
        }
    std::priority_queue<Node, std::vector<Node>, Compare> queue;
    queue.push(root);
    double incumbent = inf, closed = inf, unresolved = inf;
    int64_t unresolved_count = 0;
    auto install = [&](std::vector<double> x, const std::vector<double> &y) {
        for (size_t j = 0; j < x.size(); j++)
            if (original.types[j] != VarType::Continuous)
                x[j] = std::round(x[j]);
        auto a = verify(original, x, y);
        if (a.finite && a.primal <= options.tol && a.integrality <= options.integer_tol &&
            a.objective < incumbent) {
            incumbent = a.objective;
            out.x = std::move(x);
            out.y = y;
            out.accuracy = a;
        }
    };
    auto cutoff = [&](double bound) {
        return std::isfinite(incumbent) &&
               bound >= incumbent - options.mip_gap * std::max(1., std::abs(incumbent));
    };
    while (!queue.empty() && out.nodes < options.node_limit) {
        if (interrupted) {
            out.status = "INTERRUPTED";
            break;
        }
        if (elapsed(start) >= options.time_limit) {
            out.status = "TIME_LIMIT";
            break;
        }
        Node node = queue.top();
        queue.pop();
        if (cutoff(node.bound)) {
            closed = std::min(closed, node.bound);
            continue;
        }
        if (options.presolve && !propagate(original, node, out.bounds_tightened)) {
            out.nodes++;
            continue;
        }
        relaxation.lb = node.lb;
        relaxation.ub = node.ub;
        Options o = options;
        o.initial_x = node.x;
        o.initial_y = node.y;
        o.time_limit = std::max(0., options.time_limit - elapsed(start));
        o.verbose = false;
        auto r = solve_continuous(relaxation, o);
        out.nodes++;
        out.iterations += r.iterations;
        out.restarts += r.restarts;
        out.rejected_steps += r.rejected_steps;
        out.iteration_seconds += r.iteration_seconds;
        out.transfer_seconds += r.transfer_seconds;
        out.preprocess_seconds += r.preprocess_seconds;
        out.verification_seconds += r.verification_seconds;
        out.backend = r.backend;
        out.device_name = r.device_name;
        if (r.status == "INFEASIBLE")
            continue;
        double lower = std::max(node.bound, r.accuracy.lower_bound);
        if (cutoff(lower)) {
            closed = std::min(closed, lower);
            continue;
        }
        if (r.x.size() != original.c.size() || !r.accuracy.finite) {
            unresolved = std::min(unresolved, lower);
            unresolved_count++;
            continue;
        }
        install(r.x, r.y);
        // Round-and-repair fixes integer assignments, then uses our continuous engine.
        if (out.nodes == 1 || out.nodes % 10 == 0) {
            Model repair = relaxation;
            for (size_t j = 0; j < repair.c.size(); j++)
                if (original.types[j] != VarType::Continuous)
                    repair.lb[j] = repair.ub[j] =
                        std::clamp(std::round(r.x[j]), node.lb[j], node.ub[j]);
            Options ro = o;
            ro.initial_x = r.x;
            ro.initial_y = r.y;
            ro.iteration_limit = std::min<int64_t>(o.iteration_limit, 5000);
            ro.time_limit = std::max(0., options.time_limit - elapsed(start));
            auto rr = solve_continuous(repair, ro);
            out.iterations += rr.iterations;
            out.restarts += rr.restarts;
            out.rejected_steps += rr.rejected_steps;
            out.iteration_seconds += rr.iteration_seconds;
            out.verification_seconds += rr.verification_seconds;
            out.preprocess_seconds += rr.preprocess_seconds;
            out.transfer_seconds += rr.transfer_seconds;
            if (rr.x.size() == original.c.size())
                install(rr.x, rr.y);
        }
        if (cutoff(lower)) {
            closed = std::min(closed, lower);
            continue;
        }
        int64_t branch = -1;
        double fraction = options.integer_tol;
        for (size_t j = 0; j < r.x.size(); j++)
            if (original.types[j] != VarType::Continuous && node.lb[j] < node.ub[j]) {
                double v = std::clamp(r.x[j], node.lb[j], node.ub[j]);
                double f = std::abs(v - std::round(v));
                if (f > fraction) {
                    fraction = f;
                    branch = j;
                }
            }
        // An approximately integral primal is not an optimality proof. Keep unresolved leaves.
        if (branch < 0) {
            unresolved = std::min(unresolved, lower);
            unresolved_count++;
            continue;
        }
        double value = std::clamp(r.x[branch], node.lb[branch], node.ub[branch]);
        Node left = node, right = node;
        left.bound = right.bound = lower;
        left.depth = right.depth = node.depth + 1;
        left.x = right.x = r.x;
        left.y = right.y = r.y;
        left.ub[branch] = std::floor(value);
        right.lb[branch] = std::ceil(value);
        if (left.lb[branch] <= left.ub[branch])
            queue.push(std::move(left));
        if (right.lb[branch] <= right.ub[branch])
            queue.push(std::move(right));
    }
    out.nodes_remaining = queue.size() + unresolved_count;
    out.best_bound = std::min(closed, unresolved);
    if (!queue.empty())
        out.best_bound = std::min(out.best_bound, queue.top().bound);
    if (std::isfinite(incumbent)) {
        out.best_bound = std::min(out.best_bound, incumbent);
        out.mip_gap = std::isfinite(out.best_bound) ? std::max(0., incumbent - out.best_bound) /
                                                          std::max(1., std::abs(incumbent))
                                                    : inf;
        if (out.mip_gap <= options.mip_gap)
            out.status = "OPTIMAL";
        else if (queue.empty())
            out.status = "UNKNOWN";
        out.accuracy = verify(original, out.x, out.y);
    } else if (queue.empty()) {
        out.status = unresolved_count ? "UNKNOWN" : "INFEASIBLE";
    }
    if (out.status == "UNKNOWN")
        out.message =
            "Unresolved LP leaves retained; no unverified primal objective was used to prune";
    out.seconds = elapsed(start);
    return out;
}
} // namespace vantage
