#include "gnn.hpp"
#include <memory>
#include "internal.hpp"
#include "json.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>
namespace vantage {
namespace {
struct Node {
    std::vector<double> lb, ub, x, y;
    std::vector<int64_t> basis;
    std::string basis_fingerprint;
    std::vector<int64_t> changed;
    std::vector<double> changed_lb, changed_ub;
    double estimate = -inf;
    double bound = -inf;
    uint64_t id = 0;
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
// Conservative interval arithmetic for implied node variable bounds. Integer bounds
// are rounded inward only after enclosing the implied endpoint; continuous endpoints
// are rounded outward. These valid node-box reductions support finite dual bounds.
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
                {
                    bool integer = m.types[j] != VarType::Continuous;
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
                    if (std::isfinite(implied_low) &&
                        (!integer || std::abs(implied_low) < 0x1p52L)) {
                        double bound = integer ? double(std::ceil(implied_low))
                                               : std::nextafter(double(implied_low), -inf);
                        if (bound > node.lb[j]) {
                            node.lb[j] = bound;
                            changed = true;
                            tightened++;
                        }
                    }
                    if (std::isfinite(implied_high) &&
                        (!integer || std::abs(implied_high) < 0x1p52L)) {
                        double bound = integer ? double(std::floor(implied_high))
                                               : std::nextafter(double(implied_high), inf);
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
        return a.bound != b.bound ? a.bound > b.bound : a.id > b.id;
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
    out.method_selected = options.cuts ? "branch-and-cut" : "branch-and-bound";
    out.status = "NODE_LIMIT";
    Model relaxation = original;
    if (options.cuts) {
        out.cuts_added = add_binary_cuts(relaxation, 64);
        out.cuts_added += add_integer_lattice_cuts(relaxation, 32);
        out.cuts_added += add_mir_cuts(relaxation, 64);
    }
    std::fill(relaxation.types.begin(), relaxation.types.end(), VarType::Continuous);
    const Model root_relaxation = relaxation;
    std::unique_ptr<BranchingGnn> gnn;
    if (options.branching == "gnn")
        gnn = std::make_unique<BranchingGnn>(BranchingGnn::load(options.branching_model));
    Node root;
    root.lb = original.lb;
    root.ub = original.ub;
    root.x = options.initial_x;
    root.y = options.initial_y;
    root.basis = options.initial_basis;
    root.basis_fingerprint = options.basis_fingerprint;
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
            std::vector<int64_t>().swap(node.basis);
            node.basis_fingerprint.clear();
        }
    };
    NodeQueue queue(Compare{options.node_selection});
    pack(root, true);
    queue.push(root);
    uint64_t next_node_id = 1;
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
    // Checkpoints are trusted local solver state, not standalone mathematical proofs.
    // They preserve the open tree, unresolved/closed bounds, pseudocosts and incumbent.
    using Json = nlohmann::json;
    auto number = [](double value) -> Json {
        if (std::isnan(value))
            throw std::runtime_error("NaN checkpoint state");
        return std::isfinite(value) ? Json(value) : Json(value < 0 ? "-inf" : "+inf");
    };
    auto decode = [](const Json &value) -> double {
        if (value.is_number()) {
            double result = value.get<double>();
            if (!std::isfinite(result))
                throw std::runtime_error("Nonfinite checkpoint number");
            return result;
        }
        if (value == "-inf")
            return -inf;
        if (value == "+inf")
            return inf;
        throw std::runtime_error("Invalid checkpoint number");
    };
    auto numbers = [&](const std::vector<double> &values) {
        Json result = Json::array();
        for (double v : values)
            result.push_back(number(v));
        return result;
    };
    auto decode_numbers = [&](const Json &values) {
        std::vector<double> result;
        for (const auto &v : values)
            result.push_back(decode(v));
        return result;
    };
    Json configuration = {{"method", options.method},
                          {"device", options.device},
                          {"tol", options.tol},
                          {"mip_gap", options.mip_gap},
                          {"integer_tol", options.integer_tol},
                          {"presolve", options.presolve},
                          {"restart", options.restart},
                          {"adaptive", options.adaptive},
                          {"scaling", options.scaling},
                          {"scaling_passes", options.scaling_passes},
                          {"branching", options.branching},
                          {"node_selection", options.node_selection},
                          {"cuts", options.cuts},
                          {"primal_heuristic", options.primal_heuristic},
                          {"primal_weight", options.primal_weight},
                          {"power_iterations", options.power_iterations},
                          {"polishing", options.polishing},
                          {"iteration_limit", options.iteration_limit},
                          {"check_every", options.check_every},
                          {"gpu_indices", options.gpu_indices},
                          {"matrix_precision", options.matrix_precision},
                          {"cuda_graphs", options.cuda_graphs},
                          {"gpu_monitor", options.gpu_monitor},
                          {"gpu_presolve", options.gpu_presolve},
                          {"batch_strong_branching", options.batch_strong_branching},
                          {"threads", options.threads}};
    using Conflict = std::vector<std::pair<int64_t, int>>;
    std::vector<Conflict> conflicts;
    std::vector<std::vector<BoundLiteral>> bound_conflicts;
    struct PooledCut {
        std::vector<std::pair<int64_t, double>> terms;
        double lower, upper, efficacy = 0;
        int age = 0, uses = 0;
    };
    std::vector<PooledCut> cut_pool;
    auto append_cuts = [](Model &model, const std::vector<PooledCut> &cuts) {
        if (cuts.empty())
            return;
        size_t extra = 0;
        for (const auto &cut : cuts)
            extra += cut.terms.size();
        std::vector<Entry> entries;
        entries.reserve(model.A.value.size() + extra);
        for (int64_t i = 0; i < model.A.rows; ++i)
            for (auto k = model.A.ptr[i]; k < model.A.ptr[i + 1]; ++k)
                entries.push_back({i, model.A.index[k], model.A.value[k]});
        for (const auto &cut : cuts) {
            auto row = model.rl.size();
            for (auto [j, a] : cut.terms)
                entries.push_back({int64_t(row), j, a});
            model.rl.push_back(cut.lower);
            model.ru.push_back(cut.upper);
            model.row_names.push_back("niryukti_pool_" + std::to_string(row));
        }
        model.A = Sparse::build(model.rl.size(), model.c.size(), std::move(entries));
    };
    auto learn_bound_conflict = [&](const Node &node) {
        std::vector<BoundLiteral> clause;
        bool nonbinary = false;
        for (size_t j = 0; j < original.c.size(); ++j) {
            if (node.lb[j] > base_lb[j] && std::isfinite(node.lb[j]))
                clause.push_back({int64_t(j), true, node.lb[j]});
            if (node.ub[j] < base_ub[j] && std::isfinite(node.ub[j]))
                clause.push_back({int64_t(j), false, node.ub[j]});
            nonbinary |= original.types[j] != VarType::Binary &&
                         (node.lb[j] > base_lb[j] || node.ub[j] < base_ub[j]);
        }
        if (!nonbinary || clause.empty() || clause.size() > 256)
            return;
        auto replay = [&](const std::vector<BoundLiteral> &literals) {
            Node probe;
            probe.lb = base_lb;
            probe.ub = base_ub;
            for (auto lit : literals) {
                auto &bound = lit.lower ? probe.lb[lit.variable] : probe.ub[lit.variable];
                bound = lit.lower ? std::max(bound, lit.value) : std::min(bound, lit.value);
            }
            int64_t ignored = 0;
            return !propagate(original, probe, ignored);
        };
        // Every clause is replay-proved using original rows, independently of
        // learned cuts and relaxation limit statuses.
        if (!replay(clause))
            return;
        int trials = 0;
        for (size_t i = 0; i < clause.size() && trials < 16 && !stop_requested(options) &&
                           elapsed(start) < options.time_limit;) {
            auto reduced = clause;
            reduced.erase(reduced.begin() + i);
            ++trials;
            if (!reduced.empty() && replay(reduced))
                clause = std::move(reduced);
            else
                ++i;
        }
        std::sort(clause.begin(), clause.end());
        for (const auto &old : bound_conflicts)
            if (std::includes(clause.begin(), clause.end(), old.begin(), old.end()))
                return;
        bound_conflicts.erase(std::remove_if(bound_conflicts.begin(), bound_conflicts.end(),
                                             [&](const auto &old) {
                                                 return std::includes(old.begin(), old.end(),
                                                                      clause.begin(), clause.end());
                                             }),
                              bound_conflicts.end());
        if (bound_conflicts.size() == 128)
            bound_conflicts.erase(bound_conflicts.begin());
        bound_conflicts.push_back(std::move(clause));
        ++out.bound_conflicts_learned;
    };
    auto learn_conflict = [&](const Node &node) {
        learn_bound_conflict(node);
        Conflict conflict;
        bool nonbinary_changes = false;
        for (size_t j = 0; j < original.c.size(); ++j) {
            if (original.types[j] != VarType::Binary)
                nonbinary_changes |= node.lb[j] != base_lb[j] || node.ub[j] != base_ub[j];
            else if (node.lb[j] == node.ub[j])
                conflict.push_back({int64_t(j), int(node.lb[j])});
        }
        if (conflict.empty())
            return;
        auto replay_infeasible = [&](const Conflict &clause) {
            Node probe;
            probe.lb = base_lb;
            probe.ub = base_ub;
            for (auto [j, v] : clause)
                probe.lb[j] = probe.ub[j] = v;
            int64_t ignored = 0;
            return !propagate(original, probe, ignored);
        };
        // Nonbinary assumptions cannot enter a binary clause. Learn only if
        // original-row propagation independently proves the binary subset.
        if (nonbinary_changes && !replay_infeasible(conflict))
            return;
        // Deletion filtering produces a smaller proved clause; limited work
        // keeps explanation generation inside the solve time budget.
        int trials = 0;
        int trial_limit = original.A.value.size() <= 100000 ? 16 : 2;
        for (size_t i = 0; i < conflict.size() && trials < trial_limit &&
                           !stop_requested(options) && elapsed(start) < options.time_limit;) {
            auto reduced = conflict;
            reduced.erase(reduced.begin() + i);
            ++trials;
            if (!reduced.empty() && replay_infeasible(reduced))
                conflict = std::move(reduced);
            else
                ++i;
        }
        for (const auto &existing : conflicts)
            if (std::includes(conflict.begin(), conflict.end(), existing.begin(), existing.end()))
                return; // Existing clause already excludes this conjunction.
        conflicts.erase(std::remove_if(conflicts.begin(), conflicts.end(),
                                       [&](const auto &old) {
                                           return std::includes(old.begin(), old.end(),
                                                                conflict.begin(), conflict.end());
                                       }),
                        conflicts.end());
        if (conflicts.size() == 128)
            conflicts.erase(conflicts.begin());
        conflicts.push_back(std::move(conflict));
        out.conflicts_learned++;
    };
    auto snapshot = [&]() {
        if (options.checkpoint_path.empty())
            return;
        Json saved = {{"schema", "vantage-tree-1"},
                      {"model_fingerprint", original.fingerprint()},
                      {"configuration", configuration},
                      {"conflicts", conflicts},
                      {"incumbent", number(incumbent)},
                      {"closed", number(closed)},
                      {"unresolved", number(unresolved)},
                      {"unresolved_count", unresolved_count},
                      {"next_node_id", next_node_id},
                      {"x", out.x},
                      {"y", out.y}};
        saved["bound_conflicts"] = Json::array();
        for (const auto &clause : bound_conflicts) {
            Json literals = Json::array();
            for (auto l : clause)
                literals.push_back(
                    {{"variable", l.variable}, {"lower", l.lower}, {"value", l.value}});
            saved["bound_conflicts"].push_back(literals);
        }
        saved["cut_pool"] = Json::array();
        for (const auto &cut : cut_pool)
            saved["cut_pool"].push_back({{"terms", cut.terms},
                                         {"lower", number(cut.lower)},
                                         {"upper", number(cut.upper)},
                                         {"efficacy", cut.efficacy},
                                         {"age", cut.age},
                                         {"uses", cut.uses}});
        saved["queue"] = Json::array();
        auto copy = queue;
        while (!copy.empty()) {
            const auto &n = copy.top();
            saved["queue"].push_back({{"id", n.id},
                                      {"changed", n.changed},
                                      {"lb", numbers(n.changed_lb)},
                                      {"ub", numbers(n.changed_ub)},
                                      {"x", n.x},
                                      {"y", n.y},
                                      {"basis", n.basis},
                                      {"basis_fingerprint", n.basis_fingerprint},
                                      {"bound", number(n.bound)},
                                      {"estimate", number(n.estimate)},
                                      {"depth", n.depth},
                                      {"branch_variable", n.branch_variable},
                                      {"branch_up", n.branch_up},
                                      {"branch_distance", n.branch_distance},
                                      {"parent_objective", number(n.parent_objective)}});
            copy.pop();
        }
        saved["down"] = Json::array();
        saved["up"] = Json::array();
        for (const auto &p : down)
            saved["down"].push_back({{"mean", p.mean}, {"count", p.count}});
        for (const auto &p : up)
            saved["up"].push_back({{"mean", p.mean}, {"count", p.count}});
        saved["statistics"] = {{"nodes", out.nodes},
                               {"iterations", out.iterations},
                               {"restarts", out.restarts},
                               {"cuts_added", out.cuts_added},
                               {"cut_rounds", out.cut_rounds},
                               {"local_cuts_added", out.local_cuts_added},
                               {"bounds_tightened", out.bounds_tightened},
                               {"strong_branch_probes", out.strong_branch_probes},
                               {"pump_rounds", out.pump_rounds},
                               {"rins_calls", out.rins_calls},
                               {"heuristic_nodes", out.heuristic_nodes},
                               {"conflicts_learned", out.conflicts_learned},
                               {"conflicts_pruned", out.conflicts_pruned},
                               {"bound_conflicts_learned", out.bound_conflicts_learned},
                               {"bound_conflicts_pruned", out.bound_conflicts_pruned},
                               {"local_branching_calls", out.local_branching_calls}};
        std::filesystem::path destination(options.checkpoint_path);
        auto temporary = destination;
        temporary += ".tmp";
        {
            std::ofstream stream(temporary, std::ios::trunc);
            if (!stream)
                throw std::runtime_error("Cannot create tree checkpoint");
            stream << saved.dump(2) << '\n';
            stream.flush();
            if (!stream)
                throw std::runtime_error("Failed writing tree checkpoint");
        }
        std::filesystem::rename(temporary, destination);
    };
    if (!options.resume_path.empty()) {
        Json saved;
        std::ifstream stream(options.resume_path);
        if (!stream)
            throw std::runtime_error("Cannot open tree checkpoint");
        stream >> saved;
        if (saved.at("schema") != "vantage-tree-1" ||
            saved.at("model_fingerprint") != original.fingerprint() ||
            saved.at("configuration") != configuration)
            throw std::runtime_error("Checkpoint model, schema or solver configuration mismatch");
        for (const auto &item : saved.value("cut_pool", Json::array())) {
            PooledCut cut;
            cut.terms = item.at("terms").get<std::vector<std::pair<int64_t, double>>>();
            cut.lower = decode(item.at("lower"));
            cut.upper = decode(item.at("upper"));
            cut.efficacy = item.at("efficacy");
            cut.age = item.at("age");
            cut.uses = item.at("uses");
            if (cut.lower > cut.upper || !std::isfinite(cut.efficacy) || cut.age < 0 ||
                cut.uses < 0)
                throw std::runtime_error("Invalid checkpoint cut state");
            for (auto [j, a] : cut.terms)
                if (j < 0 || j >= int64_t(original.c.size()) || !std::isfinite(a))
                    throw std::runtime_error("Invalid checkpoint cut term");
            cut_pool.push_back(std::move(cut));
            if (cut_pool.size() > 64)
                throw std::runtime_error("Checkpoint cut pool capacity");
        }
        for (const auto &clause : saved.value("bound_conflicts", Json::array())) {
            if (clause.empty() || clause.size() > 256 || bound_conflicts.size() >= 128)
                throw std::runtime_error("Invalid checkpoint bound conflict capacity");
            std::vector<BoundLiteral> literals;
            for (const auto &item : clause) {
                BoundLiteral l{item.at("variable").get<int64_t>(), item.at("lower").get<bool>(),
                               item.at("value").get<double>()};
                if (l.variable < 0 || l.variable >= int64_t(original.c.size()) ||
                    !std::isfinite(l.value))
                    throw std::runtime_error("Invalid checkpoint bound conflict literal");
                literals.push_back(l);
            }
            std::sort(literals.begin(), literals.end());
            bound_conflicts.push_back(std::move(literals));
        }
        conflicts = saved.at("conflicts").get<std::vector<Conflict>>();
        if (conflicts.size() > 128)
            throw std::runtime_error("Checkpoint conflict pool too large");
        for (const auto &conflict : conflicts) {
            if (conflict.empty())
                throw std::runtime_error("Empty checkpoint conflict");
            std::set<int64_t> variables;
            for (auto [j, v] : conflict)
                if (j < 0 || j >= int64_t(original.c.size()) ||
                    original.types[j] != VarType::Binary || (v != 0 && v != 1) ||
                    !variables.insert(j).second)
                    throw std::runtime_error("Malformed checkpoint binary conflict");
        }
        incumbent = decode(saved.at("incumbent"));
        closed = decode(saved.at("closed"));
        unresolved = decode(saved.at("unresolved"));
        next_node_id = saved.at("next_node_id").get<uint64_t>();
        unresolved_count = saved.at("unresolved_count").get<int64_t>();
        if (unresolved_count < 0)
            throw std::runtime_error("Invalid unresolved checkpoint count");
        out.x = saved.at("x").get<std::vector<double>>();
        out.y = saved.at("y").get<std::vector<double>>();
        if (std::isfinite(incumbent)) {
            auto a = verify(original, out.x, out.y);
            if (!a.finite || a.primal > options.tol || a.integrality > options.integer_tol ||
                std::abs(a.objective - incumbent) > options.tol * (1 + std::abs(incumbent)))
                throw std::runtime_error("Checkpoint incumbent fails original-model verification");
            out.accuracy = a;
        } else if (!out.x.empty() || !out.y.empty())
            throw std::runtime_error("Checkpoint has vectors without an incumbent");
        queue = NodeQueue(Compare{options.node_selection});
        for (const auto &item : saved.at("queue")) {
            Node n;
            n.id = item.at("id").get<uint64_t>();
            n.changed = item.at("changed").get<std::vector<int64_t>>();
            n.changed_lb = decode_numbers(item.at("lb"));
            n.changed_ub = decode_numbers(item.at("ub"));
            n.basis = item.at("basis").get<std::vector<int64_t>>();
            n.basis_fingerprint = item.at("basis_fingerprint");
            n.x = item.at("x").get<std::vector<double>>();
            n.y = item.at("y").get<std::vector<double>>();
            n.bound = decode(item.at("bound"));
            n.estimate = decode(item.at("estimate"));
            n.depth = item.at("depth");
            n.branch_variable = item.at("branch_variable");
            n.branch_up = item.at("branch_up");
            n.branch_distance = item.at("branch_distance");
            n.parent_objective = decode(item.at("parent_objective"));
            if (n.changed.size() != n.changed_lb.size() ||
                n.changed.size() != n.changed_ub.size() || n.depth < 0 || n.branch_variable < -1 ||
                n.branch_variable >= int64_t(original.c.size()) ||
                !std::isfinite(n.branch_distance) || n.branch_distance < 0 ||
                (!n.x.empty() && n.x.size() != original.c.size()) ||
                (!n.y.empty() && n.y.size() != root_relaxation.rl.size()))
                throw std::runtime_error("Malformed checkpoint node");
            for (size_t k = 0; k < n.changed.size(); ++k) {
                auto j = n.changed[k];
                if (j < 0 || j >= int64_t(base_lb.size()) || std::isnan(n.changed_lb[k]) ||
                    std::isnan(n.changed_ub[k]) || n.changed_lb[k] < base_lb[j] ||
                    n.changed_ub[k] > base_ub[j] || n.changed_lb[k] > n.changed_ub[k])
                    throw std::runtime_error("Invalid checkpoint node bounds");
            }
            for (double v : n.x)
                if (!std::isfinite(v))
                    throw std::runtime_error("Invalid checkpoint primal");
            for (double v : n.y)
                if (!std::isfinite(v))
                    throw std::runtime_error("Invalid checkpoint dual");
            queue.push(std::move(n));
        }
        for (auto [target, key] : {std::pair{&down, "down"}, std::pair{&up, "up"}}) {
            const auto &items = saved.at(key);
            if (items.size() != target->size())
                throw std::runtime_error("Checkpoint pseudocost dimension mismatch");
            for (size_t j = 0; j < items.size(); ++j) {
                (*target)[j].mean = items[j].at("mean");
                (*target)[j].count = items[j].at("count");
                if (!std::isfinite((*target)[j].mean) || (*target)[j].mean < 0 ||
                    (*target)[j].count < 0)
                    throw std::runtime_error("Invalid checkpoint pseudocost");
            }
        }
        const auto &stats = saved.at("statistics");
        out.nodes = stats.at("nodes");
        out.iterations = stats.at("iterations");
        out.restarts = stats.at("restarts");
        out.cuts_added = stats.at("cuts_added");
        out.cut_rounds = stats.at("cut_rounds");
        out.local_cuts_added = stats.at("local_cuts_added");
        out.bounds_tightened = stats.at("bounds_tightened");
        out.strong_branch_probes = stats.at("strong_branch_probes");
        out.pump_rounds = stats.at("pump_rounds");
        out.rins_calls = stats.at("rins_calls");
        out.heuristic_nodes = stats.at("heuristic_nodes");
        out.conflicts_learned = stats.at("conflicts_learned");
        out.conflicts_pruned = stats.at("conflicts_pruned");
        out.bound_conflicts_learned = stats.value("bound_conflicts_learned", int64_t(0));
        out.bound_conflicts_pruned = stats.value("bound_conflicts_pruned", int64_t(0));
        out.local_branching_calls = stats.at("local_branching_calls");
        if (out.nodes < 0 || out.iterations < 0)
            throw std::runtime_error("Negative checkpoint statistics");
    }
    while (!queue.empty() && out.nodes < options.node_limit) {
        if (out.nodes % options.checkpoint_nodes == 0)
            snapshot();
        if (stop_requested(options)) {
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
        if (!propagate_bound_conflicts(bound_conflicts, original.types, node.lb, node.ub,
                                       out.bounds_tightened)) {
            ++out.bound_conflicts_pruned;
            ++out.nodes;
            continue;
        }
        bool conflict_pruned =
            !propagate_binary_conflicts(conflicts, node.lb, node.ub, out.bounds_tightened);
        if (conflict_pruned) {
            out.conflicts_pruned++;
            out.nodes++;
            continue;
        }
        if (options.gpu_presolve) {
            auto old_lb = node.lb, old_ub = node.ub;
            if (!cuda_propagate_integer_bounds(original, node.lb, node.ub)) {
                learn_conflict(node);
                out.nodes++;
                continue;
            }
            for (size_t j = 0; j < node.lb.size(); ++j)
                out.bounds_tightened += (old_lb[j] != node.lb[j]) + (old_ub[j] != node.ub[j]);
        }
        if (options.presolve && !propagate(original, node, out.bounds_tightened)) {
            learn_conflict(node);
            out.nodes++;
            continue;
        }
        relaxation = root_relaxation;
        std::vector<PooledCut> active_cuts;
        if (options.cuts) {
            std::stable_sort(cut_pool.begin(), cut_pool.end(),
                             [](const auto &a, const auto &b) { return a.efficacy > b.efficacy; });
            for (size_t i = 0; i < std::min<size_t>(16, cut_pool.size()); ++i) {
                active_cuts.push_back(cut_pool[i]);
                cut_pool[i].uses++;
            }
        }
        if (options.cuts) {
            // sum_{v=0} x_j + sum_{v=1}(1-x_j) >= 1.
            // These rows are globally valid because learned conjunctions are
            // checked only after certified infeasibility, never a limit status.
            for (const auto &conflict : conflicts) {
                PooledCut cut;
                cut.lower = 1;
                cut.upper = inf;
                for (auto [j, v] : conflict) {
                    cut.terms.push_back({j, v ? -1. : 1.});
                    cut.lower -= v;
                }
                active_cuts.push_back(std::move(cut));
            }
        }
        append_cuts(relaxation, active_cuts);
        relaxation.lb = node.lb;
        relaxation.ub = node.ub;
        Options o = options;
        // Node bounds already passed GPU propagation; do not repeat it in its LP.
        o.gpu_presolve = false;
        o.checkpoint_path.clear();
        o.resume_path.clear();
        o.initial_x = node.x;
        o.initial_basis = node.basis;
        o.basis_fingerprint = node.basis_fingerprint;
        o.initial_y = node.y;
        if (!o.initial_y.empty())
            o.initial_y.resize(relaxation.rl.size(), 0);
        o.time_limit = std::max(0., options.time_limit - elapsed(start));
        o.verbose = false;
        auto r = solve_continuous(relaxation, o);
        if (out.relaxation_method_selected.empty())
            out.relaxation_method_selected = r.method_selected;
        // Bounds used to derive these cuts apply only to this node. They are discarded
        // on exit; only their certified relaxation lower bound is inherited by children.
        for (int round = 0; options.cuts && round < 2 && !stop_requested(options) &&
                            elapsed(start) < options.time_limit && r.status != "INFEASIBLE" &&
                            r.accuracy.finite && r.x.size() == original.c.size();
             ++round) {
            Model separated = relaxation;
            separated.types = original.types;
            int added = add_integer_lattice_cuts(separated, 8, &r.x);
            added += add_mir_cuts(separated, 16, &r.x);
            if (!added)
                break;
            account_heuristic(r);
            out.cuts_added += added;
            out.local_cuts_added += added;
            out.cut_rounds++;
            std::fill(separated.types.begin(), separated.types.end(), VarType::Continuous);
            relaxation = std::move(separated);
            o.initial_x = r.x;
            o.initial_y = r.y;
            o.initial_y.resize(relaxation.rl.size(), 0);
            o.time_limit = std::max(0., options.time_limit - elapsed(start));
            r = solve_continuous(relaxation, o);
        }
        if (options.cuts && r.accuracy.finite && r.x.size() == original.c.size()) {
            for (auto &cut : cut_pool) {
                long double activity = 0, norm = 0;
                for (auto [j, a] : cut.terms) {
                    activity += (long double)a * r.x[j];
                    norm += (long double)a * a;
                }
                cut.efficacy = double(std::max({0.L, (long double)cut.lower - activity,
                                                activity - (long double)cut.upper}) /
                                      std::max(1.L, std::sqrt(norm)));
                cut.age = cut.efficacy > 1e-7 ? 0 : cut.age + 1;
            }
            cut_pool.erase(std::remove_if(cut_pool.begin(), cut_pool.end(),
                                          [](const auto &cut) { return cut.age > 10; }),
                           cut_pool.end());
            // Global candidates are derived only from root bounds, never node bounds.
            auto global = root_relaxation;
            global.types = original.types;
            size_t first = global.rl.size();
            add_integer_lattice_cuts(global, 8, &r.x);
            add_mir_cuts(global, 16, &r.x);
            for (size_t row = first; row < global.rl.size(); ++row) {
                PooledCut cut;
                cut.lower = global.rl[row];
                cut.upper = global.ru[row];
                long double activity = 0, norm = 0;
                for (auto k = global.A.ptr[row]; k < global.A.ptr[row + 1]; ++k) {
                    auto j = global.A.index[k];
                    double a = global.A.value[k];
                    cut.terms.push_back({j, a});
                    activity += (long double)a * r.x[j];
                    norm += (long double)a * a;
                }
                cut.efficacy = double(std::max({0.L, (long double)cut.lower - activity,
                                                activity - (long double)cut.upper}) /
                                      std::max(1.L, std::sqrt(norm)));
                bool duplicate = false;
                for (const auto &old : cut_pool)
                    if (old.terms == cut.terms && old.lower == cut.lower && old.upper == cut.upper)
                        duplicate = true;
                if (!duplicate) {
                    if (cut_pool.size() == 64)
                        cut_pool.erase(std::min_element(
                            cut_pool.begin(), cut_pool.end(),
                            [](const auto &a, const auto &b) { return a.efficacy < b.efficacy; }));
                    cut_pool.push_back(std::move(cut));
                    out.cuts_added++;
                }
            }
        }
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
        if (r.status == "INFEASIBLE") {
            learn_conflict(node);
            continue;
        }
        double lower = std::max(node.bound, r.accuracy.lower_bound);
        if (options.verbose)
            std::cerr << "node=" << node.id << " relaxation_status=" << r.status
                      << " objective=" << r.accuracy.objective
                      << " safe_bound=" << r.accuracy.lower_bound << " kkt=" << r.accuracy.kkt
                      << " x=" << r.x.size() << " y=" << r.y.size() << '\n';
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
            ro.initial_basis = r.basis;
            ro.basis_fingerprint = r.basis_fingerprint;
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
            for (int round = 0;
                 round < 8 && !stop_requested(options) && elapsed(start) < options.time_limit;
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
            out.nodes < options.node_limit && !stop_requested(options) &&
            elapsed(start) < options.time_limit) {
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
                no.checkpoint_path.clear();
                no.resume_path.clear();
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
        if (std::isfinite(incumbent) && (out.nodes == 1 || out.nodes % 20 == 0) &&
            (options.primal_heuristic == "local" || options.primal_heuristic == "all") &&
            out.nodes < options.node_limit && !stop_requested(options) &&
            elapsed(start) < options.time_limit) {
            Model neighborhood = relaxation;
            neighborhood.types = original.types;
            std::vector<Entry> entries;
            for (int64_t i = 0; i < neighborhood.A.rows; ++i)
                for (auto k = neighborhood.A.ptr[i]; k < neighborhood.A.ptr[i + 1]; ++k)
                    entries.push_back({i, neighborhood.A.index[k], neighborhood.A.value[k]});
            int ones = 0, binaries = 0;
            auto row = neighborhood.A.rows;
            for (size_t j = 0; j < original.c.size(); ++j)
                if (original.types[j] == VarType::Binary) {
                    bool one = out.x[j] > .5;
                    ones += one;
                    binaries++;
                    entries.push_back({row, int64_t(j), one ? -1. : 1.});
                }
            if (binaries) {
                neighborhood.rl.push_back(-inf);
                neighborhood.ru.push_back(std::min(5, std::max(1, binaries / 4)) - ones);
                std::string name = "vantage_local_branching";
                while (std::find(neighborhood.row_names.begin(), neighborhood.row_names.end(),
                                 name) != neighborhood.row_names.end())
                    name += "_";
                neighborhood.row_names.push_back(name);
                neighborhood.A = Sparse::build(neighborhood.rl.size(), neighborhood.c.size(),
                                               std::move(entries));
                Options no = o;
                no.primal_heuristic = "repair";
                no.cuts = false;
                no.checkpoint_path.clear();
                no.resume_path.clear();
                no.initial_x = out.x;
                no.initial_y.assign(neighborhood.rl.size(), 0);
                no.node_limit = std::min<int64_t>(20, options.node_limit - out.nodes);
                no.time_limit = std::max(0., options.time_limit - elapsed(start));
                auto candidate = solve_mip(neighborhood, no);
                account_heuristic(candidate);
                out.local_branching_calls++;
                out.nodes += candidate.nodes;
                out.heuristic_nodes += candidate.nodes;
                if (candidate.x.size() == original.c.size())
                    install(candidate.x, candidate.y);
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
            if (options.verbose)
                std::cerr << "unresolved_node=" << node.id << " relaxation_status=" << r.status
                          << " kkt=" << r.accuracy.kkt << " primal=" << r.accuracy.primal
                          << " objective=" << r.accuracy.objective << " bound=" << lower
                          << " message=" << r.message << '\n';
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
            std::vector<Result> batched;
            std::vector<std::array<int, 2>> batch_index(candidates.size(), {-1, -1});
            if (options.batch_strong_branching && r.status == "OPTIMAL") {
                std::vector<std::vector<double>> lows, upper_boxes;
                for (size_t rank = 0; rank < std::min<size_t>(4, candidates.size()); ++rank) {
                    auto j = candidates[rank].second;
                    double v = std::clamp(r.x[j], node.lb[j], node.ub[j]);
                    for (int direction = 0; direction < 2; ++direction) {
                        if ((direction ? up[j] : down[j]).count >= 2)
                            continue;
                        batch_index[rank][direction] = int(lows.size());
                        lows.push_back(node.lb);
                        upper_boxes.push_back(node.ub);
                        if (direction)
                            lows.back()[j] = std::ceil(v);
                        else
                            upper_boxes.back()[j] = std::floor(v);
                    }
                }
                if (!lows.empty()) {
                    Options po = o;
                    po.initial_x = r.x;
                    po.initial_y = r.y;
                    po.iteration_limit = std::min<int64_t>(o.iteration_limit, 1000);
                    po.time_limit = std::max(0., options.time_limit - elapsed(start));
                    batched = cuda_batch_relaxations(relaxation, lows, upper_boxes, po);
                }
            }
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
                        if (cost.count >= 2 || stop_requested(options) ||
                            elapsed(start) >= options.time_limit)
                            continue;
                        Model probe = relaxation;
                        if (direction)
                            probe.lb[j] = std::ceil(v);
                        else
                            probe.ub[j] = std::floor(v);
                        Options po = o;
                        po.initial_x = r.x;
                        po.initial_basis = r.basis;
                        po.basis_fingerprint = r.basis_fingerprint;
                        po.initial_y = r.y;
                        po.iteration_limit = std::min<int64_t>(o.iteration_limit, 1000);
                        po.time_limit = std::max(0., options.time_limit - elapsed(start));
                        auto pr = batch_index[rank][direction] >= 0
                                      ? batched.at(batch_index[rank][direction])
                                      : solve_continuous(probe, po);
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
        if (gnn) {
            // Learned scores over the bipartite constraint/variable graph of this node LP.
            Model typed = relaxation;
            typed.types = original.types;
            auto graph = BranchingGnn::structure(typed);
            BranchingGnn::features(graph, typed, r.x, r.y, options.integer_tol);
            if (!graph.candidates.empty()) {
                auto score = gnn->scores(graph);
                auto best = std::max_element(score.begin(), score.end()) - score.begin();
                if (std::isfinite(score[best]))
                    branch = graph.candidates[best];
            }
        }
        double value = std::clamp(r.x[branch], node.lb[branch], node.ub[branch]);
        Node left = node, right = node;
        left.id = next_node_id++;
        right.id = next_node_id++;
        left.bound = right.bound = lower;
        left.depth = right.depth = node.depth + 1;
        left.x = right.x = r.x;
        left.basis = right.basis = r.basis;
        left.basis_fingerprint = right.basis_fingerprint = r.basis_fingerprint;
        left.y = right.y = r.y;
        // Local row multipliers must never be assigned to another node's rows.
        left.y.resize(root_relaxation.rl.size(), 0);
        right.y.resize(root_relaxation.rl.size(), 0);
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
    snapshot();
    out.seconds = elapsed(start);
    return out;
}
} // namespace vantage
