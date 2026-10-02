#include "farkas.hpp"
#include "internal.hpp"
#include "json.hpp"
#include "vantage/analysis.hpp"
#include <fstream>
#include <iomanip>
#include <map>
#include <queue>
#include <set>
#include <sstream>
// Global optimization of bilinear programs (pooling/blending with quality x volume terms).
//   min c'x + sum_p g_p x_a(p) x_b(p)
//   s.t. rl <= A x + sum_p H_ip x_a(p) x_b(p) <= ru,  lb <= x <= ub.
// Each product w = x_a x_b is replaced by its McCormick (1976) convex/concave envelope
// over the node box. Spatial branch-and-bound (Al-Khayyal & Falk 1983; Tawarmalani &
// Sahinidis 2002) bisects the variable that causes the largest envelope violation.
// Lower bounds come from the verifier's outward-rounded Lagrangian bound of each LP
// relaxation; upper bounds come from exactly re-evaluated feasible points produced by an
// alternating LP heuristic (fix one side of every product, solve the remaining LP).
namespace vantage {
namespace {
using json = nlohmann::json;
struct Product {
    int64_t a, b;
};
struct Term {
    int64_t pair;
    double coef;
};
struct Problem {
    std::string name = "bilinear";
    double sense = 1, offset = 0;
    std::vector<std::string> names, row_names;
    std::vector<double> lb, ub, c, rl, ru;
    std::vector<std::vector<std::pair<int64_t, double>>> linear; // per row
    std::vector<std::vector<Term>> bilinear;                      // per row
    std::vector<Term> objective;
    std::vector<Product> pairs;
    std::vector<double> weight; // total |coefficient| of each pair, for branching
};

double bound_value(const json &j, double fallback) {
    if (j.is_null())
        return fallback;
    if (j.is_string()) {
        auto s = j.get<std::string>();
        if (s == "inf" || s == "+inf" || s == "infinity")
            return inf;
        if (s == "-inf" || s == "-infinity")
            return -inf;
        throw std::runtime_error("Invalid bound: " + s);
    }
    return j.get<double>();
}

Problem parse(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open " + path);
    json j;
    in >> j;
    Problem p;
    p.name = j.value("name", "bilinear");
    auto sense = j.value("sense", "min");
    if (sense != "min" && sense != "max" && sense != "minimize" && sense != "maximize")
        throw std::runtime_error("sense must be min or max");
    p.sense = (sense == "max" || sense == "maximize") ? -1 : 1;
    std::map<std::string, int64_t> id;
    for (const auto &v : j.at("variables")) {
        auto name = v.at("name").get<std::string>();
        if (id.count(name))
            throw std::runtime_error("Duplicate variable " + name);
        if (v.value("type", "continuous") != "continuous")
            throw std::runtime_error("Bilinear global solve supports continuous variables only");
        id[name] = int64_t(p.names.size());
        p.names.push_back(name);
        p.lb.push_back(bound_value(v.value("lb", json(0)), -inf));
        p.ub.push_back(bound_value(v.value("ub", json()), inf));
        if (p.lb.back() > p.ub.back())
            throw std::runtime_error("Invalid bounds for " + name);
    }
    auto n = p.names.size();
    p.c.assign(n, 0.);
    std::map<std::pair<int64_t, int64_t>, int64_t> pair_id;
    auto var = [&](const json &v) {
        auto s = v.get<std::string>();
        if (!id.count(s))
            throw std::runtime_error("Unknown variable: " + s);
        return id[s];
    };
    auto product = [&](const json &t) {
        if (!t.is_array() || t.size() != 3)
            throw std::runtime_error("bilinear terms must be [variable, variable, coefficient]");
        auto a = var(t[0]), b = var(t[1]);
        if (a > b)
            std::swap(a, b);
        auto key = std::make_pair(a, b);
        if (!pair_id.count(key)) {
            pair_id[key] = int64_t(p.pairs.size());
            p.pairs.push_back({a, b});
            p.weight.push_back(0);
        }
        double coef = t[2].get<double>();
        p.weight[pair_id[key]] += std::abs(coef);
        return Term{pair_id[key], coef};
    };
    const auto &obj = j.at("objective");
    if (obj.contains("linear")) {
        const auto &l = obj.at("linear");
        if (l.is_array()) {
            if (l.size() != n)
                throw std::runtime_error("objective.linear array length must match variables");
            for (size_t k = 0; k < n; ++k)
                p.c[k] = l[k].get<double>();
        } else
            for (auto it = l.begin(); it != l.end(); ++it)
                p.c[var(json(it.key()))] += it.value().get<double>();
    }
    if (obj.contains("bilinear"))
        for (const auto &t : obj.at("bilinear"))
            p.objective.push_back(product(t));
    p.offset = obj.value("offset", 0.);
    for (const auto &row : j.at("constraints")) {
        p.row_names.push_back(row.value("name", "r" + std::to_string(p.rl.size())));
        p.rl.push_back(bound_value(row.value("lb", json()), -inf));
        p.ru.push_back(bound_value(row.value("ub", json()), inf));
        std::vector<std::pair<int64_t, double>> lin;
        if (row.contains("coefficients"))
            for (auto it = row.at("coefficients").begin(); it != row.at("coefficients").end();
                 ++it)
                lin.push_back({var(json(it.key())), it.value().get<double>()});
        std::vector<Term> bil;
        if (row.contains("bilinear"))
            for (const auto &t : row.at("bilinear"))
                bil.push_back(product(t));
        p.linear.push_back(lin);
        p.bilinear.push_back(bil);
    }
    // Canonical minimization.
    for (auto &v : p.c)
        v *= p.sense;
    for (auto &t : p.objective)
        t.coef *= p.sense;
    p.offset *= p.sense;
    for (auto &pr : p.pairs)
        for (auto v : {pr.a, pr.b})
            if (!std::isfinite(p.lb[v]) || !std::isfinite(p.ub[v]))
                throw std::runtime_error("Variable '" + p.names[v] +
                                         "' appears in a product and needs finite bounds");
    return p;
}

// Exact evaluation in long double; returns canonical objective and max violation.
std::pair<double, double> evaluate(const Problem &p, const std::vector<double> &x) {
    long double obj = p.offset;
    for (size_t j = 0; j < x.size(); ++j)
        obj += (long double)p.c[j] * x[j];
    for (auto &t : p.objective)
        obj += (long double)t.coef * x[p.pairs[t.pair].a] * x[p.pairs[t.pair].b];
    double violation = 0;
    for (size_t j = 0; j < x.size(); ++j)
        violation = std::max({violation, (p.lb[j] - x[j]) / (1 + std::abs(p.lb[j])),
                              (x[j] - p.ub[j]) / (1 + std::abs(p.ub[j]))});
    for (size_t i = 0; i < p.rl.size(); ++i) {
        long double a = 0;
        for (auto &[j, v] : p.linear[i])
            a += (long double)v * x[j];
        for (auto &t : p.bilinear[i])
            a += (long double)t.coef * x[p.pairs[t.pair].a] * x[p.pairs[t.pair].b];
        if (std::isfinite(p.rl[i]))
            violation = std::max(violation, double((p.rl[i] - a) / (1 + std::abs(p.rl[i]))));
        if (std::isfinite(p.ru[i]))
            violation = std::max(violation, double((a - p.ru[i]) / (1 + std::abs(p.ru[i]))));
    }
    return {double(obj), violation};
}

Model base_model(const Problem &p, size_t extra_columns) {
    Model m;
    m.name = p.name;
    size_t n = p.names.size() + extra_columns;
    m.c.assign(n, 0.);
    m.q.assign(n, 0.);
    m.types.assign(n, VarType::Continuous);
    m.names = p.names;
    for (size_t k = 0; k < extra_columns; ++k)
        m.names.push_back("__w" + std::to_string(k));
    m.lb = p.lb;
    m.ub = p.ub;
    m.lb.resize(n, -inf);
    m.ub.resize(n, inf);
    m.offset = p.offset;
    return m;
}

// LP relaxation with McCormick envelopes on box [lo, hi].
Model relaxation(const Problem &p, const std::vector<double> &lo, const std::vector<double> &hi) {
    const size_t n = p.names.size(), np = p.pairs.size();
    Model m = base_model(p, np);
    for (size_t j = 0; j < n; ++j) {
        m.lb[j] = lo[j];
        m.ub[j] = hi[j];
        m.c[j] = p.c[j];
    }
    for (auto &t : p.objective)
        m.c[n + t.pair] += t.coef;
    std::vector<Entry> e;
    for (size_t i = 0; i < p.rl.size(); ++i) {
        for (auto &[j, v] : p.linear[i])
            e.push_back({int64_t(i), j, v});
        for (auto &t : p.bilinear[i])
            e.push_back({int64_t(i), int64_t(n + t.pair), t.coef});
        m.rl.push_back(p.rl[i]);
        m.ru.push_back(p.ru[i]);
        m.row_names.push_back(p.row_names[i]);
    }
    // Outward nudge keeps floating-point envelope coefficients valid.
    auto loosen = [](double v, bool lower) {
        double slack = 1e-12 * (1 + std::abs(v));
        return lower ? v - slack : v + slack;
    };
    int64_t row = int64_t(p.rl.size());
    for (size_t k = 0; k < np; ++k) {
        auto a = p.pairs[k].a, b = p.pairs[k].b;
        double la = lo[a], ua = hi[a], lb = lo[b], ub = hi[b];
        int64_t w = int64_t(n + k);
        double corners[] = {la * lb, la * ub, ua * lb, ua * ub};
        m.lb[w] = loosen(*std::min_element(corners, corners + 4), true);
        m.ub[w] = loosen(*std::max_element(corners, corners + 4), false);
        // w >= lb*x_a + la*x_b - la*lb ; w >= ub*x_a + ua*x_b - ua*ub
        // w <= ub*x_a + la*x_b - la*ub ; w <= lb*x_a + ua*x_b - ua*lb
        struct Cut {
            double ca, cb, rhs;
            bool lower;
        } cuts[] = {{lb, la, -la * lb, true},
                    {ub, ua, -ua * ub, true},
                    {ub, la, -la * ub, false},
                    {lb, ua, -ua * lb, false}};
        for (auto &cut : cuts) {
            e.push_back({row, w, 1});
            if (a == b)
                e.push_back({row, a, -(cut.ca + cut.cb)});
            else {
                e.push_back({row, a, -cut.ca});
                e.push_back({row, b, -cut.cb});
            }
            m.rl.push_back(cut.lower ? loosen(cut.rhs, true) : -inf);
            m.ru.push_back(cut.lower ? inf : loosen(cut.rhs, false));
            m.row_names.push_back("__mccormick_" + std::to_string(k) + "_" + std::to_string(row));
            ++row;
        }
    }
    m.A = Sparse::build(row, int64_t(m.c.size()), std::move(e));
    return m;
}

// LP obtained by fixing every variable with fixed[j] at value[j]; products become linear.
Model restriction(const Problem &p, const std::vector<double> &lo, const std::vector<double> &hi,
                  const std::vector<char> &fixed, const std::vector<double> &value) {
    const size_t n = p.names.size();
    Model m = base_model(p, 0);
    for (size_t j = 0; j < n; ++j) {
        double v = std::clamp(value[j], lo[j], hi[j]);
        m.lb[j] = fixed[j] ? v : lo[j];
        m.ub[j] = fixed[j] ? v : hi[j];
        m.c[j] = p.c[j];
    }
    auto linearize = [&](const Term &t, auto emit, double &constant) {
        auto a = p.pairs[t.pair].a, b = p.pairs[t.pair].b;
        if (fixed[a] && fixed[b])
            constant += t.coef * m.lb[a] * m.lb[b];
        else if (fixed[a])
            emit(b, t.coef * m.lb[a]);
        else if (fixed[b])
            emit(a, t.coef * m.lb[b]);
        else
            throw std::runtime_error("restriction leaves a product unfixed");
    };
    double obj_constant = 0;
    for (auto &t : p.objective)
        linearize(t, [&](int64_t j, double v) { m.c[j] += v; }, obj_constant);
    m.offset += obj_constant;
    std::vector<Entry> e;
    for (size_t i = 0; i < p.rl.size(); ++i) {
        std::map<int64_t, double> row;
        for (auto &[j, v] : p.linear[i])
            row[j] += v;
        double constant = 0;
        for (auto &t : p.bilinear[i])
            linearize(t, [&](int64_t j, double v) { row[j] += v; }, constant);
        for (auto &[j, v] : row)
            if (v != 0)
                e.push_back({int64_t(i), j, v});
        m.rl.push_back(p.rl[i] - constant);
        m.ru.push_back(p.ru[i] - constant);
        m.row_names.push_back(p.row_names[i]);
    }
    m.A = Sparse::build(int64_t(p.rl.size()), int64_t(n), std::move(e));
    return m;
}

struct Node {
    std::vector<double> lo, hi;
    double bound;
    int64_t depth;
    bool operator<(const Node &o) const { return bound > o.bound; }
};
std::string text(double v) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(4) << v;
    return s.str();
}
} // namespace

std::string global_solve_json(const std::string &path, const Options &options) {
    auto start = Clock::now();
    auto p = parse(path);
    const size_t n = p.names.size();
    Options lp_options;
    lp_options.device = "cpu";
    lp_options.tol = std::min(options.tol, 1e-8);
    lp_options.time_limit = std::max(1., options.time_limit);
    double abs_tol = 1e-6, rel_tol = options.mip_gap;
    // Simplex first; fall back to the automatic portfolio when it is inconclusive.
    auto solve_lp = [&](const Model &lp) {
        Result r;
        for (const char *method : {"simplex", "auto"}) {
            Options o = lp_options;
            o.method = method;
            if (o.method == "simplex" && lp.A.rows > simplex_row_limit())
                continue;
            r = solve(lp, o);
            if (r.status == "OPTIMAL" || r.status == "INFEASIBLE")
                break;
        }
        return r;
    };
    // Greedy vertex cover of the product graph: fixing the cover makes every product linear.
    std::vector<char> cover(n, 0), other(n, 0);
    {
        std::vector<char> covered(p.pairs.size(), 0);
        while (true) {
            std::vector<int64_t> degree(n, 0);
            bool left = false;
            for (size_t k = 0; k < p.pairs.size(); ++k)
                if (!covered[k]) {
                    left = true;
                    ++degree[p.pairs[k].a];
                    if (p.pairs[k].b != p.pairs[k].a)
                        ++degree[p.pairs[k].b];
                }
            if (!left)
                break;
            auto v = std::max_element(degree.begin(), degree.end()) - degree.begin();
            cover[v] = 1;
            for (size_t k = 0; k < p.pairs.size(); ++k)
                if (p.pairs[k].a == v || p.pairs[k].b == v)
                    covered[k] = 1;
        }
        for (auto &pr : p.pairs)
            for (auto v : {pr.a, pr.b})
                if (!cover[v])
                    other[v] = 1;
    }
    double incumbent = inf;
    std::vector<double> best;
    std::vector<double> local_values;
    int64_t heuristic_solves = 0, relaxation_solves = 0, nodes = 0, pruned_infeasible = 0,
            pruned_bound = 0, unresolved = 0;
    double unresolved_bound = inf, pruned_floor = inf;
    bool safe_bounds = true;
    auto consider = [&](const std::vector<double> &x) {
        auto [obj, violation] = evaluate(p, x);
        if (violation <= 1e-7 && obj < incumbent - 1e-9) {
            incumbent = obj;
            best = x;
            local_values.push_back(obj == 0 ? 0. : p.sense * obj);
        }
    };
    // Alternating LP heuristic: fix the cover, solve; fix the complement, solve; repeat.
    auto heuristic = [&](const std::vector<double> &lo, const std::vector<double> &hi,
                         std::vector<double> point) {
        double previous = inf;
        for (int round = 0; round < 6; ++round) {
            auto &fixed = round % 2 == 0 ? cover : other;
            if (std::none_of(fixed.begin(), fixed.end(), [](char f) { return f; }))
                break;
            auto lp = restriction(p, lo, hi, fixed, point);
            ++heuristic_solves;
            auto r = solve_lp(lp);
            if (r.status != "OPTIMAL")
                break;
            point = r.x;
            consider(point);
            auto value = evaluate(p, point).first;
            if (round > 0 && value > previous - 1e-9 * (1 + std::abs(previous)))
                break;
            previous = std::min(previous, value);
        }
    };
    std::priority_queue<Node> open;
    open.push({p.lb, p.ub, -inf, 0});
    std::vector<double> root_x;
    double root_bound = -inf;
    std::string stop = "COMPLETE";
    while (!open.empty()) {
        if (elapsed(start) > options.time_limit) {
            stop = "TIME_LIMIT";
            break;
        }
        if (nodes >= options.node_limit) {
            stop = "NODE_LIMIT";
            break;
        }
        Node node = open.top();
        if (node.bound >= incumbent - std::max(abs_tol, rel_tol * std::abs(incumbent))) {
            pruned_bound += int64_t(open.size());
            pruned_floor = std::min(pruned_floor, node.bound);
            while (!open.empty())
                open.pop();
            break;
        }
        open.pop();
        ++nodes;
        auto relax = relaxation(p, node.lo, node.hi);
        ++relaxation_solves;
        auto r = solve_lp(relax);
        if (r.status == "INFEASIBLE" && certify_infeasible(relax, r) > 0) {
            ++pruned_infeasible;
            continue;
        }
        if (r.status != "OPTIMAL") {
            ++unresolved;
            unresolved_bound = std::min(unresolved_bound, node.bound);
            continue;
        }
        double bound = r.accuracy.lower_bound;
        if (!std::isfinite(bound)) {
            safe_bounds = false;
            bound = r.accuracy.objective - 1e-9 * (1 + std::abs(r.accuracy.objective));
        }
        bound = std::max(bound, node.bound);
        std::vector<double> x(r.x.begin(), r.x.begin() + n);
        if (nodes == 1) {
            root_x = x;
            root_bound = bound;
        }
        consider(x);
        heuristic(node.lo, node.hi, x);
        if (bound >= incumbent - std::max(abs_tol, rel_tol * std::abs(incumbent))) {
            ++pruned_bound;
            pruned_floor = std::min(pruned_floor, bound);
            continue;
        }
        // Branch on the product with the largest weighted envelope violation.
        double worst = 0;
        int64_t pick = -1;
        for (size_t k = 0; k < p.pairs.size(); ++k) {
            auto a = p.pairs[k].a, b = p.pairs[k].b;
            double gap = std::abs(r.x[n + k] - x[a] * x[b]) * (1 + p.weight[k]);
            if (gap > worst) {
                worst = gap;
                pick = int64_t(k);
            }
        }
        if (pick < 0 || worst <= 1e-9) {
            // Relaxation point satisfies every product: it is feasible and optimal here.
            ++pruned_bound;
            pruned_floor = std::min(pruned_floor, bound);
            continue;
        }
        auto a = p.pairs[pick].a, b = p.pairs[pick].b;
        auto width = [&](int64_t v) {
            double root = p.ub[v] - p.lb[v];
            return root > 0 ? (node.hi[v] - node.lo[v]) / root : 0.;
        };
        auto v = width(a) >= width(b) ? a : b;
        double w = node.hi[v] - node.lo[v];
        if (w <= 1e-9 * (1 + std::abs(node.hi[v]))) {
            ++unresolved;
            unresolved_bound = std::min(unresolved_bound, bound);
            continue;
        }
        double split = std::clamp(x[v], node.lo[v] + .1 * w, node.hi[v] - .1 * w);
        Node left{node.lo, node.hi, bound, node.depth + 1}, right = left;
        left.hi[v] = split;
        right.lo[v] = split;
        open.push(std::move(left));
        open.push(std::move(right));
    }
    // Global bound: weakest of open, unresolved and bound-pruned regions; infeasible
    // regions contribute nothing.
    double lower = std::min(unresolved_bound, pruned_floor);
    while (!open.empty()) {
        lower = std::min(lower, open.top().bound);
        open.pop();
    }
    bool exhausted = stop == "COMPLETE";
    lower = std::min(lower, incumbent);
    bool have = std::isfinite(incumbent);
    double gap = have && std::isfinite(lower) ? std::max(0., incumbent - lower) : inf;
    bool certified = have && safe_bounds && gap <= std::max(abs_tol, rel_tol * std::abs(incumbent));
    json badge;
    std::string status;
    if (!have && exhausted && unresolved == 0 && pruned_infeasible > 0) {
        status = "INFEASIBLE";
        badge = {{"state", "CERTIFIED_INFEASIBLE"}, {"color", "green"},
                 {"meaning", "Every region of the domain has a verified Farkas certificate"}};
    } else if (certified) {
        status = "GLOBAL_OPTIMAL";
        badge = {{"state", "CERTIFIED_GLOBAL"}, {"color", "green"},
                 {"meaning", "The rigorous relaxation bound meets the best feasible plan"}};
    } else if (have && std::isfinite(lower)) {
        status = "BOUNDED";
        badge = {{"state", "BOUNDED"}, {"color", "amber"},
                 {"meaning", "Feasible plan with a proven limit on how much better any plan "
                             "can be"},
                 {"unverified_gap", gap}};
    } else if (have) {
        status = "LOCAL_ONLY";
        badge = {{"state", "LOCAL_ONLY"}, {"color", "red"},
                 {"meaning", "Feasible plan without a valid global bound"}};
    } else {
        status = "NO_SOLUTION";
        badge = {{"state", "NO_SOLUTION"}, {"color", "red"},
                 {"meaning", "No feasible plan found and infeasibility not proven"}};
    }
    // Original-sense reporting: for maximization the bound is a ceiling.
    auto original = [&](double v) { return std::isfinite(v) ? json(p.sense * v) : json(); };
    json report = {{"analysis", "BILINEAR_GLOBAL"},
                   {"model", p.name},
                   {"status", status},
                   {"badge", badge},
                   {"sense", p.sense > 0 ? "min" : "max"},
                   {"objective", original(incumbent)},
                   {p.sense > 0 ? "lower_bound" : "upper_bound", original(lower)},
                   {"absolute_gap", std::isfinite(gap) ? json(gap) : json()},
                   {"relative_gap",
                    std::isfinite(gap) ? json(gap / std::max(1., std::abs(incumbent))) : json()},
                   {"root_relaxation_bound", original(root_bound)},
                   {"stop_reason", stop},
                   {"nodes", nodes},
                   {"relaxation_solves", relaxation_solves},
                   {"heuristic_solves", heuristic_solves},
                   {"pruned_by_bound", pruned_bound},
                   {"pruned_infeasible", pruned_infeasible},
                   {"unresolved_nodes", unresolved},
                   {"products", p.pairs.size()},
                   {"improving_solutions", local_values},
                   {"seconds", elapsed(start)}};
    if (have) {
        auto [obj, violation] = evaluate(p, best);
        json values = json::object();
        for (size_t j = 0; j < n; ++j)
            values[p.names[j]] = best[j];
        report["solution"] = values;
        report["verification"] = {
            {"recomputed_objective", p.sense * obj},
            {"max_scaled_violation", violation},
            {"feasible", violation <= 1e-7},
            {"method", "independent long-double evaluation of every linear and bilinear row"}};
        std::ostringstream s;
        if (status == "GLOBAL_OPTIMAL")
            s << "Certified global: no plan can beat " << text(p.sense * incumbent)
              << "; the relaxation bound matches it within tolerance.";
        else if (status == "BOUNDED")
            s << "Bounded: best plan " << text(p.sense * incumbent) << "; no plan can exceed "
              << text(p.sense * lower) << ". At most " << text(gap)
              << " objective units remain unverified.";
        else
            s << "Local only: plan " << text(p.sense * incumbent)
              << " is feasible but no valid global bound is available.";
        report["summary"] = s.str();
    }
    report["scope"] = "Bilinear products of bounded continuous variables. Bounds come from "
                      "outward-rounded Lagrangian bounds of McCormick LP relaxations; envelope "
                      "coefficients are nudged outward by 1e-12 relative.";
    if (!safe_bounds)
        report["warning"] = "At least one relaxation lacked a finite verified dual bound";
    return report.dump(2);
}
} // namespace vantage
