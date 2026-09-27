#include "internal.hpp"
#include <queue>
#include <set>
#include <utility>
namespace vantage {
namespace {
struct Node {
    std::vector<double> lb, ub, x, y;
    std::vector<int64_t> changed;
    std::vector<double> changed_lb, changed_ub;
    double estimate = -inf;
    double bound = -inf;
    int depth = 0;
    int64_t branch_variable = -1;
    bool branch_up = false;
    double branch_distance = 0, parent_objective = inf;
};
struct Pseudocost {
    double mean = 0;
    int64_t count = 0;
    void observe(double parent, double child, double distance) {
        if (!std::isfinite(parent) || !std::isfinite(child) || distance <= 0)
            return;
        double value = std::max(0., child - parent) / distance;
        if (!std::isfinite(value))
            return;
        mean += (value - mean) / double(++count);
    }
    double predict(double distance, double fallback) const {
        return distance * (count ? mean : fallback);
    }
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
    std::string policy = "best-bound";
    bool operator()(const Node &a, const Node &b) const {
        if (policy == "depth-first" && a.depth != b.depth)
            return a.depth < b.depth;
        if (policy == "best-estimate" && a.estimate != b.estimate)
            return a.estimate > b.estimate;
        return a.bound > b.bound;
    }
};
struct NodeQueue : std::priority_queue<Node, std::vector<Node>, Compare> {
    using std::priority_queue<Node, std::vector<Node>, Compare>::priority_queue;
    double lower_bound() const {
        double bound = inf;
        for (const auto &node : this->c)
            bound = std::min(bound, node.bound);
        return bound;
    }
};
} // namespace
Result solve_mip(const Model &original, const Options &options) {
    auto start = Clock::now();
    Result out;
    out.method_selected = options.cuts ? "branch-and-bound-with-root-cuts" : "branch-and-bound";
    out.status = "NODE_LIMIT";
    Model relaxation = original;
    if (options.cuts) {
        out.cuts_added = add_binary_cuts(relaxation, 64);
        out.cuts_added += add_mir_cuts(relaxation, 64);
    }
    std::fill(relaxation.types.begin(), relaxation.types.end(), VarType::Continuous);
    Node root;
    root.lb = original.lb;
    root.ub = original.ub;
    root.x = options.initial_x;
    root.y = options.initial_y;
    if (!root.y.empty())
        root.y.resize(relaxation.rl.size(), 0);
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
    const auto base_lb = root.lb, base_ub = root.ub;
    auto pack = [&](Node &node, bool preserve_warm = false) {
        node.changed.clear();
        node.changed_lb.clear();
        node.changed_ub.clear();
        for (size_t j = 0; j < base_lb.size(); ++j)
            if (node.lb[j] != base_lb[j] || node.ub[j] != base_ub[j]) {
                node.changed.push_back(j);
                node.changed_lb.push_back(node.lb[j]);
                node.changed_ub.push_back(node.ub[j]);
            }
        std::vector<double>().swap(node.lb);
        std::vector<double>().swap(node.ub);
        if (base_lb.size() > 10000 && !preserve_warm) {
            std::vector<double>().swap(node.x);
            std::vector<double>().swap(node.y);
        }
    };
    NodeQueue queue(Compare{options.node_selection});
    pack(root, true);
    queue.push(root);
    double incumbent = inf, closed = inf, unresolved = inf;
    int64_t unresolved_count = 0;
    size_t cost_count = options.branching == "reliability" ? original.c.size() : 0;
    std::vector<Pseudocost> down(cost_count), up(cost_count);
    auto install = [&](std::vector<double> x, const std::vector<double> &y) {
        for (size_t j = 0; j < x.size(); j++)
            if (original.types[j] != VarType::Continuous)
                x[j] = std::round(x[j]);
        // Cut multipliers belong to the integer relaxation, not original rows.
        auto original_y = y;
        original_y.resize(original.rl.size(), 0);
        auto a = verify(original, x, original_y);
        if (a.finite && a.primal <= options.tol && a.integrality <= options.integer_tol &&
            a.objective < incumbent) {
            incumbent = a.objective;
            out.x = std::move(x);
            out.y = std::move(original_y);
            out.accuracy = a;
        }
    };
    auto cutoff = [&](double bound) {
        return std::isfinite(incumbent) &&
               bound >= incumbent - options.mip_gap * std::max(1., std::abs(incumbent));
    };
    auto account_heuristic = [&](const Result &h) {
        out.iterations += h.iterations;
        out.restarts += h.restarts;
        out.rejected_steps += h.rejected_steps;
        out.weight_updates += h.weight_updates;
        out.polishing_iterations += h.polishing_iterations;
        out.polishing_attempts += h.polishing_attempts;
        out.preprocess_seconds += h.preprocess_seconds;
        out.transfer_seconds += h.transfer_seconds;
        out.iteration_seconds += h.iteration_seconds;
        out.verification_seconds += h.verification_seconds;
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
        node.lb = base_lb;
        node.ub = base_ub;
        for (size_t k = 0; k < node.changed.size(); ++k) {
            node.lb[node.changed[k]] = node.changed_lb[k];
            node.ub[node.changed[k]] = node.changed_ub[k];
        }
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
        out.weight_updates += r.weight_updates;
        out.polishing_iterations += r.polishing_iterations;
        out.polishing_attempts += r.polishing_attempts;
        out.iteration_seconds += r.iteration_seconds;
        out.transfer_seconds += r.transfer_seconds;
        out.preprocess_seconds += r.preprocess_seconds;
        out.verification_seconds += r.verification_seconds;
        out.backend = r.backend;
        out.device_name = r.device_name;
        out.device_reason = r.device_reason;
        out.estimated_gpu_bytes = r.estimated_gpu_bytes;
        out.monitor_checks += r.monitor_checks;
        out.host_candidate_checks += r.host_candidate_checks;
        out.skipped_candidate_checks += r.skipped_candidate_checks;
        out.graph_execution = r.graph_execution;
        out.gpu_index_bits = r.gpu_index_bits;
        out.matrix_precision = r.matrix_precision;
        if (options.branching == "reliability" && node.branch_variable >= 0 &&
            r.status == "OPTIMAL") {
            auto &cost = node.branch_up ? up[node.branch_variable] : down[node.branch_variable];
            cost.observe(node.parent_objective, r.accuracy.objective, node.branch_distance);
        }
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
            out.weight_updates += rr.weight_updates;
            out.polishing_iterations += rr.polishing_iterations;
            out.polishing_attempts += rr.polishing_attempts;
            out.iteration_seconds += rr.iteration_seconds;
            out.verification_seconds += rr.verification_seconds;
            out.preprocess_seconds += rr.preprocess_seconds;
            out.transfer_seconds += rr.transfer_seconds;
            if (rr.x.size() == original.c.size())
                install(rr.x, rr.y);
        }
        if (out.nodes == 1 && !std::isfinite(incumbent) &&
            (options.primal_heuristic == "pump" || options.primal_heuristic == "all")) {
            auto point = r.x;
            std::set<std::vector<double>> visited;
            for (int round = 0; round < 8 && !interrupted && elapsed(start) < options.time_limit;
                 ++round) {
                auto target = point;
                std::vector<double> key;
                for (size_t j = 0; j < target.size(); ++j)
                    if (original.types[j] != VarType::Continuous) {
                        target[j] = std::clamp(std::round(target[j]), node.lb[j], node.ub[j]);
                        key.push_back(target[j]);
                    }
                if (!visited.insert(key).second) {
                    bool changed = false;
                    for (size_t t = 0; t < target.size(); ++t) {
                        size_t j = (t + size_t(round)) % target.size();
                        if (original.types[j] == VarType::Continuous)
                            continue;
                        if (target[j] + 1 <= node.ub[j] && target[j] + 1 != target[j]) {
                            target[j] += 1;
                            changed = true;
                            break;
                        }
                        if (target[j] - 1 >= node.lb[j] && target[j] - 1 != target[j]) {
                            target[j] -= 1;
                            changed = true;
                            break;
                        }
                    }
                    if (!changed)
                        break;
                }
                Model source = relaxation;
                source.types = original.types;
                auto projection = distance_projection_model(source, target);
                Options po = o;
                po.initial_x = point;
                for (size_t j = 0; j < point.size(); ++j)
                    if (original.types[j] != VarType::Continuous)
                        po.initial_x.push_back(std::abs(point[j] - target[j]));
                po.initial_y.assign(projection.rl.size(), 0);
                po.polishing = false;
                po.iteration_limit = std::min<int64_t>(o.iteration_limit, 2000);
                po.time_limit = std::max(0., options.time_limit - elapsed(start));
                auto projected = solve_continuous(projection, po);
                account_heuristic(projected);
                out.pump_rounds++;
                if (!projected.accuracy.finite || projected.x.size() < original.c.size())
                    break;
                point.assign(projected.x.begin(), projected.x.begin() + original.c.size());
                install(point, std::vector<double>(original.rl.size(), 0));
                if (std::isfinite(incumbent))
                    break;
            }
        }
        if (std::isfinite(incumbent) && (out.nodes == 1 || out.nodes % 10 == 0) &&
            (options.primal_heuristic == "rins" || options.primal_heuristic == "all") &&
            out.nodes < options.node_limit && !interrupted && elapsed(start) < options.time_limit) {
            Model neighborhood = relaxation;
            neighborhood.types = original.types;
            size_t fixed = 0;
            for (size_t j = 0; j < original.c.size(); ++j)
                if (original.types[j] != VarType::Continuous && node.lb[j] < node.ub[j] &&
                    std::abs(r.x[j] - out.x[j]) <= options.integer_tol && out.x[j] >= node.lb[j] &&
                    out.x[j] <= node.ub[j]) {
                    neighborhood.lb[j] = neighborhood.ub[j] = out.x[j];
                    fixed++;
                }
            if (fixed) {
                Options no = o;
                no.primal_heuristic = "repair";
                no.cuts = false;
                no.initial_x = r.x;
                no.initial_y = r.y;
                no.node_limit = std::min<int64_t>(20, options.node_limit - out.nodes);
                no.time_limit = std::max(0., options.time_limit - elapsed(start));
                auto nr = solve_mip(neighborhood, no);
                account_heuristic(nr);
                out.rins_calls++;
                out.nodes += nr.nodes;
                out.heuristic_nodes += nr.nodes;
                if (nr.x.size() == original.c.size())
                    install(nr.x, nr.y);
                // Neighborhood bounds concern only its restricted feasible set.
                // They never enter the global tree's lower bound or cutoff logic.
            }
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
        if (options.branching == "reliability") {
            std::vector<std::pair<double, size_t>> candidates;
            for (size_t j = 0; j < r.x.size(); j++)
                if (original.types[j] != VarType::Continuous && node.lb[j] < node.ub[j]) {
                    double v = std::clamp(r.x[j], node.lb[j], node.ub[j]);
                    double f = std::abs(v - std::round(v));
                    if (f > options.integer_tol)
                        candidates.emplace_back(-f, j);
                }
            std::sort(candidates.begin(), candidates.end());
            double best_score = -1;
            for (size_t rank = 0; rank < candidates.size(); rank++) {
                auto j = candidates[rank].second;
                double v = std::clamp(r.x[j], node.lb[j], node.ub[j]);
                double distance[2] = {v - std::floor(v), std::ceil(v) - v};
                // Probe only a bounded shortlist. Limited/failed probes are never proofs
                // and never contribute observations to the reliability count.
                if (rank < 4 && r.status == "OPTIMAL")
                    for (int direction = 0; direction < 2; direction++) {
                        auto &cost = direction ? up[j] : down[j];
                        if (cost.count >= 2 || interrupted || elapsed(start) >= options.time_limit)
                            continue;
                        Model probe = relaxation;
                        if (direction)
                            probe.lb[j] = std::ceil(v);
                        else
                            probe.ub[j] = std::floor(v);
                        Options po = o;
                        po.initial_x = r.x;
                        po.initial_y = r.y;
                        po.iteration_limit = std::min<int64_t>(o.iteration_limit, 1000);
                        po.time_limit = std::max(0., options.time_limit - elapsed(start));
                        auto pr = solve_continuous(probe, po);
                        out.strong_branch_probes++;
                        out.iterations += pr.iterations;
                        out.restarts += pr.restarts;
                        out.rejected_steps += pr.rejected_steps;
                        out.weight_updates += pr.weight_updates;
                        out.polishing_iterations += pr.polishing_iterations;
                        out.polishing_attempts += pr.polishing_attempts;
                        out.preprocess_seconds += pr.preprocess_seconds;
                        out.iteration_seconds += pr.iteration_seconds;
                        out.transfer_seconds += pr.transfer_seconds;
                        out.verification_seconds += pr.verification_seconds;
                        if (pr.status == "OPTIMAL")
                            cost.observe(r.accuracy.objective, pr.accuracy.objective,
                                         distance[direction]);
                    }
                double fallback = std::max(1., std::abs(original.c[j]));
                double d = down[j].predict(distance[0], fallback);
                double u = up[j].predict(distance[1], fallback);
                double score = .9 * std::min(d, u) + .1 * std::max(d, u);
                if (score > best_score) {
                    best_score = score;
                    branch = j;
                }
            }
        }
        double value = std::clamp(r.x[branch], node.lb[branch], node.ub[branch]);
        Node left = node, right = node;
        left.bound = right.bound = lower;
        left.depth = right.depth = node.depth + 1;
        left.x = right.x = r.x;
        left.y = right.y = r.y;
        left.branch_variable = right.branch_variable = branch;
        left.branch_up = false;
        right.branch_up = true;
        left.branch_distance = value - std::floor(value);
        right.branch_distance = std::ceil(value) - value;
        left.parent_objective = right.parent_objective =
            r.status == "OPTIMAL" ? r.accuracy.objective : inf;
        left.ub[branch] = std::floor(value);
        right.lb[branch] = std::ceil(value);
        const double fallback = std::max(1., std::abs(original.c[branch]));
        left.estimate = r.accuracy.objective +
                        (down.empty() ? fallback * left.branch_distance
                                      : down[branch].predict(left.branch_distance, fallback));
        right.estimate = r.accuracy.objective +
                         (up.empty() ? fallback * right.branch_distance
                                     : up[branch].predict(right.branch_distance, fallback));
        if (left.lb[branch] <= left.ub[branch]) {
            pack(left);
            queue.push(std::move(left));
        }
        if (right.lb[branch] <= right.ub[branch]) {
            pack(right);
            queue.push(std::move(right));
        }
    }
    out.nodes_remaining = queue.size() + unresolved_count;
    out.best_bound = std::min(closed, unresolved);
    if (!queue.empty())
        out.best_bound = std::min(out.best_bound, queue.lower_bound());
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
