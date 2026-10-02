#include "dense_lu.hpp"
#include "internal.hpp"
#include "json.hpp"
#include "vantage/analysis.hpp"
#include <algorithm>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
// Basis-exact post-optimal analysis for LPs in bounded form
//   min c'x  s.t.  Ax - s = 0,  lb <= x <= ub,  rl <= s <= ru.
// A basis B (m columns of [A  -I]) is reconstructed from the optimal vertex, repaired by
// degenerate Bland pivots until it is dual feasible, then used for classical ranging
// (Dantzig 1963; Chvatal 1983 ch. 10; Bertsimas & Tsitsiklis 1997 sec. 5.1-5.2).
namespace vantage {
namespace {
using json = nlohmann::json;
constexpr int64_t row_limit = 2000, objective_ranging_limit = 1500, pivot_row_limit = 600;
enum class State { Basic, Lower, Upper, Fixed, Free };
const char *state_name(State s) {
    switch (s) {
    case State::Basic: return "basic";
    case State::Lower: return "at_lower";
    case State::Upper: return "at_upper";
    case State::Fixed: return "fixed";
    default: return "free_nonbasic";
    }
}
json number(double v) {
    return std::isfinite(v) ? json(v) : json();
}
std::string text(double v) {
    if (!std::isfinite(v))
        return v > 0 ? "+infinity" : "-infinity";
    std::ostringstream s;
    s << std::setprecision(6) << (std::abs(v) < 1e-12 ? 0. : v);
    return s.str();
}

struct Basis {
    const Model &m;
    Sparse at;
    int64_t n, rows;
    std::vector<double> lo, hi, cost, value;
    std::vector<State> state;
    std::vector<int64_t> basic, position;
    MixedPrecisionSolver solver;
    bool mixed = true;

    explicit Basis(const Model &model)
        : m(model), at(model.A.transpose()), n(int64_t(model.c.size())), rows(model.A.rows) {
        lo = m.lb;
        hi = m.ub;
        lo.insert(lo.end(), m.rl.begin(), m.rl.end());
        hi.insert(hi.end(), m.ru.begin(), m.ru.end());
        cost = m.c;
        cost.resize(n + rows, 0.);
        position.assign(n + rows, -1);
    }
    void column(int64_t k, std::vector<double> &out) const {
        out.assign(rows, 0.);
        if (k < n)
            for (auto p = at.ptr[k]; p < at.ptr[k + 1]; ++p)
                out[at.index[p]] = at.value[p];
        else
            out[k - n] = -1;
    }
    double dot(int64_t k, const std::vector<double> &v) const {
        if (k >= n)
            return -v[k - n];
        long double s = 0;
        for (auto p = at.ptr[k]; p < at.ptr[k + 1]; ++p)
            s += (long double)at.value[p] * v[at.index[p]];
        return double(s);
    }
    void factor() {
        std::vector<double> dense(size_t(rows) * rows), col;
        for (int64_t r = 0; r < rows; ++r) {
            column(basic[r], col);
            std::copy(col.begin(), col.end(), dense.begin() + r * rows);
        }
        if (!solver.factor(std::move(dense), rows, mixed))
            throw std::runtime_error("Reconstructed basis is singular");
    }
    void recompute_basic_values() {
        std::vector<double> rhs(rows, 0.), col;
        for (int64_t k = 0; k < n + rows; ++k)
            if (state[k] != State::Basic && value[k] != 0) {
                column(k, col);
                for (int64_t i = 0; i < rows; ++i)
                    rhs[i] -= col[i] * value[k];
            }
        auto xb = solver.solve(rhs);
        for (int64_t r = 0; r < rows; ++r)
            value[basic[r]] = xb[r];
    }
    std::vector<double> duals() {
        std::vector<double> cb(rows);
        for (int64_t r = 0; r < rows; ++r)
            cb[r] = cost[basic[r]];
        return solver.solve(cb, true);
    }
    double reduced(int64_t k, const std::vector<double> &pi) const {
        return cost[k] - dot(k, pi);
    }
};

json failure(const std::string &status, const std::string &message) {
    return {{"analysis", "LP_SENSITIVITY"}, {"status", status}, {"message", message}};
}
std::string failure_text(const std::string &status, const std::string &message) {
    return failure(status, message).dump(2);
}
// Optimal basis shared by sensitivity analysis and LP differentiation.
struct Optimum {
    Model lp;
    std::unique_ptr<Basis> basis;
    std::vector<double> pi;
    bool fixed_integers = false, dual_feasible = false, pivot_limited = false;
    int64_t pivots = 0, flips = 0;
    double dtol = 0;
};
// Returns an empty string on success, otherwise a failure report.
std::string optimal_basis(const Model &source, const Options &options,
                          const std::vector<double> *given, Optimum &out) {
    Options o = options;
    o.device = "cpu";
    // Basis analysis needs a vertex: simplex first, then the automatic portfolio.
    auto solve_vertex = [&](const Model &model) {
        Result r;
        for (const char *method : {"simplex", "auto"}) {
            o.method = method;
            if (o.method == "simplex" && model.A.rows > simplex_row_limit())
                continue;
            r = solve(model, o);
            if (r.status == "OPTIMAL")
                break;
        }
        return r;
    };
    std::vector<double> x, hint;
    Model &lp = out.lp;
    lp = source;
    bool fixed_integers = false;
    if (given)
        x = *given;
    else {
        auto r = solve_vertex(source);
        if (r.status != "OPTIMAL")
            return failure_text(r.status, "Sensitivity analysis requires an optimal solution: " +
                                         r.message);
        x = r.x;
        hint = r.y;
    }
    if (x.size() != source.c.size())
        return failure_text("INVALID_INPUT", "Solution dimension does not match the model");
    if (source.is_mip()) {
        // Integer decisions are frozen at the incumbent; analysis describes the continuous
        // recourse LP, which is what planners re-price in practice.
        for (size_t j = 0; j < lp.c.size(); ++j)
            if (lp.types[j] != VarType::Continuous) {
                lp.lb[j] = lp.ub[j] = std::round(x[j]);
                lp.types[j] = VarType::Continuous;
            }
        fixed_integers = true;
        auto r = solve_vertex(lp);
        if (r.status != "OPTIMAL")
            return failure_text(r.status, "Fixed-integer LP did not solve to optimality");
        x = r.x;
        hint = r.y;
    }
    if (lp.A.rows > row_limit)
        return failure_text("UNSUPPORTED", "Dense basis analysis is limited to " +
                                          std::to_string(row_limit) + " rows");

    out.basis = std::make_unique<Basis>(lp);
    Basis &b = *out.basis;
    const auto n = b.n, rows = b.rows, total = n + rows;
    auto activity = lp.A.multiply(x);
    b.value = x;
    b.value.insert(b.value.end(), activity.begin(), activity.end());
    b.state.assign(total, State::Basic);
    // Candidate ordering: interior/free columns must be basic; at-bound columns follow by
    // the magnitude of their solver-reported reduced cost (complementary slackness).
    std::vector<double> d_hint(total, 0.);
    if (hint.size() == size_t(rows)) {
        auto aty = b.at.multiply(hint);
        for (int64_t j = 0; j < n; ++j)
            d_hint[j] = std::abs(lp.c[j] + aty[j]);
        for (int64_t i = 0; i < rows; ++i)
            d_hint[n + i] = std::abs(hint[i]);
    }
    std::vector<int64_t> interior, bounded, fixed;
    for (int64_t k = 0; k < total; ++k) {
        double l = b.lo[k], u = b.hi[k], v = b.value[k];
        if (l == u) {
            b.state[k] = State::Fixed;
            b.value[k] = l;
            fixed.push_back(k);
        } else if (std::isfinite(l) && std::abs(v - l) <= 1e-7 * (1 + std::abs(l))) {
            b.state[k] = State::Lower;
            b.value[k] = l;
            bounded.push_back(k);
        } else if (std::isfinite(u) && std::abs(v - u) <= 1e-7 * (1 + std::abs(u))) {
            b.state[k] = State::Upper;
            b.value[k] = u;
            bounded.push_back(k);
        } else
            interior.push_back(k);
    }
    std::stable_sort(bounded.begin(), bounded.end(),
                     [&](int64_t a, int64_t c) { return d_hint[a] < d_hint[c]; });
    // Greedy rank-revealing column selection by Gaussian elimination.
    std::vector<std::vector<double>> eliminated;
    std::vector<int64_t> pivots;
    std::vector<double> col;
    auto try_add = [&](int64_t k) {
        b.column(k, col);
        double scale = 0;
        for (double v : col)
            scale = std::max(scale, std::abs(v));
        for (size_t e = 0; e < eliminated.size(); ++e) {
            double f = col[pivots[e]];
            if (f != 0)
                for (int64_t i = 0; i < rows; ++i)
                    col[i] -= f * eliminated[e][i];
        }
        int64_t p = 0;
        for (int64_t i = 1; i < rows; ++i)
            if (std::abs(col[i]) > std::abs(col[p]))
                p = i;
        if (rows == 0 || std::abs(col[p]) <= 1e-9 * std::max(1., scale))
            return false;
        double inverse = 1 / col[p];
        for (auto &v : col)
            v *= inverse;
        eliminated.push_back(col);
        pivots.push_back(p);
        b.position[k] = int64_t(b.basic.size());
        b.basic.push_back(k);
        return true;
    };
    int64_t rejected_interior = 0;
    for (auto k : interior)
        if (int64_t(b.basic.size()) == rows || !try_add(k))
            ++rejected_interior;
    if (rejected_interior)
        return failure_text("NON_VERTEX",
                       "Solution has " + std::to_string(rejected_interior) +
                           " interior columns outside any basis; re-solve with --method simplex");
    for (auto group : {&bounded, &fixed})
        for (auto k : *group) {
            if (int64_t(b.basic.size()) == rows)
                break;
            if (try_add(k))
                b.state[k] = State::Basic;
        }
    eliminated.clear();
    eliminated.shrink_to_fit();
    if (int64_t(b.basic.size()) != rows)
        return failure_text("NUMERICAL_ERROR", "Could not complete a nonsingular basis");
    for (auto k : interior)
        b.state[k] = State::Basic;
    for (int64_t k = 0; k < total; ++k)
        if (b.state[k] != State::Basic && !std::isfinite(b.lo[k]) && !std::isfinite(b.hi[k]))
            b.state[k] = State::Free;
    b.mixed = o.matrix_precision != "fp64-only";
    b.factor();
    b.recompute_basic_values();

    // Degenerate basis repair: bounded primal simplex with Bland's rule. On a degenerate
    // optimal vertex the steps are zero, so the plan is unchanged and only the basis moves.
    double cmax = 1;
    for (double v : lp.c)
        cmax = std::max(cmax, std::abs(v));
    const double dtol = 1e-9 * cmax;
    int64_t pivot_count = 0, flips = 0;
    const int64_t max_pivots = 20 * rows + 100;
    bool dual_feasible = false, pivot_limited = false;
    std::vector<double> pi;
    while (true) {
        pi = b.duals();
        int64_t entering = -1;
        double direction = 0;
        for (int64_t k = 0; k < total && entering < 0; ++k) {
            if (b.state[k] == State::Basic || b.state[k] == State::Fixed)
                continue;
            double d = b.reduced(k, pi);
            if (b.state[k] == State::Lower && d < -dtol)
                entering = k, direction = 1;
            else if (b.state[k] == State::Upper && d > dtol)
                entering = k, direction = -1;
            else if (b.state[k] == State::Free && std::abs(d) > dtol)
                entering = k, direction = d < 0 ? 1 : -1;
        }
        if (entering < 0) {
            dual_feasible = true;
            break;
        }
        if (pivot_count >= max_pivots || rows > pivot_row_limit) {
            pivot_limited = true;
            break;
        }
        b.column(entering, col);
        auto w = b.solver.solve(col);
        double step = b.hi[entering] - b.lo[entering];
        int64_t leaving = -1;
        double leaving_delta = 0;
        for (int64_t r = 0; r < rows; ++r) {
            double delta = -direction * w[r];
            if (std::abs(delta) < 1e-11)
                continue;
            auto k = b.basic[r];
            double limit = delta < 0 ? (b.value[k] - b.lo[k]) / -delta
                                     : (b.hi[k] - b.value[k]) / delta;
            if (!std::isfinite(limit))
                continue;
            limit = std::max(limit, 0.);
            if (limit < step - 1e-12 ||
                (leaving >= 0 && std::abs(limit - step) <= 1e-12 && k < b.basic[leaving])) {
                step = limit;
                leaving = r;
                leaving_delta = delta;
            }
        }
        if (!std::isfinite(step))
            return failure_text("UNBOUNDED", "Ray found while repairing the basis");
        b.value[entering] += direction * step;
        for (int64_t r = 0; r < rows; ++r)
            b.value[b.basic[r]] -= direction * w[r] * step;
        if (leaving < 0) {
            b.state[entering] = direction > 0 ? State::Upper : State::Lower;
            b.value[entering] = direction > 0 ? b.hi[entering] : b.lo[entering];
            ++flips;
        } else {
            auto k = b.basic[leaving];
            b.state[k] = leaving_delta < 0 ? State::Lower : State::Upper;
            if (b.lo[k] == b.hi[k])
                b.state[k] = State::Fixed;
            b.value[k] = leaving_delta < 0 ? b.lo[k] : b.hi[k];
            b.position[k] = -1;
            b.basic[leaving] = entering;
            b.position[entering] = leaving;
            b.state[entering] = State::Basic;
            b.factor();
            b.recompute_basic_values();
        }
        ++pivot_count;
    }
    out.fixed_integers = fixed_integers;
    out.dual_feasible = dual_feasible;
    out.pivot_limited = pivot_limited;
    out.pivots = pivot_count;
    out.flips = flips;
    out.dtol = dtol;
    out.pi = pi;
    return "";
}
} // namespace

std::string sensitivity_json(const Model &source, const Options &options,
                             const std::vector<double> *given) {
    auto start = Clock::now();
    if (source.is_qp())
        return failure("UNSUPPORTED",
                       "Basis ranging applies to linear objectives; QP sensitivity is not "
                       "implemented")
            .dump(2);
    Optimum optimum;
    optimum.lp = source;
    auto problem = optimal_basis(source, options, given, optimum);
    if (!problem.empty())
        return problem;
    const Model &lp = optimum.lp;
    Basis &b = *optimum.basis;
    const auto n = b.n, rows = b.rows, total = n + rows;
    const bool fixed_integers = optimum.fixed_integers, dual_feasible = optimum.dual_feasible,
               pivot_limited = optimum.pivot_limited;
    const int64_t pivot_count = optimum.pivots, flips = optimum.flips;
    const double dtol = optimum.dtol;
    std::vector<double> pi = optimum.pi, col;
    double primal_violation = 0;
    for (int64_t k = 0; k < total; ++k) {
        double v = b.value[k];
        primal_violation = std::max({primal_violation, std::isfinite(b.lo[k]) ? b.lo[k] - v : 0.,
                                     std::isfinite(b.hi[k]) ? v - b.hi[k] : 0.});
    }
    std::vector<double> d(total, 0.);
    for (int64_t k = 0; k < total; ++k)
        if (b.state[k] != State::Basic)
            d[k] = b.reduced(k, pi);
    // Original-space verification uses the solver convention y = -pi.
    std::vector<double> xs(b.value.begin(), b.value.begin() + n), y(rows);
    for (int64_t i = 0; i < rows; ++i)
        y[i] = -pi[i];
    auto accuracy = verify(lp, xs, y);
    const double sense = lp.sense;
    auto original_range = [&](double low, double high) {
        return sense > 0 ? json::array({number(low), number(high)})
                         : json::array({number(-high), number(-low)});
    };
    auto improves = [&](double change) { return sense > 0 ? change < 0 : change > 0; };

    // Constraint ranging.
    json constraints = json::array();
    struct Binding {
        int64_t row;
        double price, low, high, bound;
        bool upper, equality;
    };
    std::vector<Binding> binding;
    for (int64_t i = 0; i < rows; ++i) {
        int64_t k = n + i;
        double act = b.value[k];
        json row = {{"name", lp.row_names[i]}, {"activity", act},
                    {"lower", number(lp.rl[i])},  {"upper", number(lp.ru[i])},
                    {"status", state_name(b.state[k])}};
        json lower_range, upper_range;
        double price = 0;
        if (b.state[k] == State::Basic) {
            if (std::isfinite(lp.ru[i]))
                upper_range = json::array({act, json()});
            if (std::isfinite(lp.rl[i]))
                lower_range = json::array({json(), act});
            double slack = std::min(std::isfinite(lp.ru[i]) ? lp.ru[i] - act : inf,
                                    std::isfinite(lp.rl[i]) ? act - lp.rl[i] : inf);
            // A basic row at its limit is degenerate: zero price, one-sided ranges.
            row["status"] = slack <= 1e-9 * (1 + std::abs(act)) ? "degenerate_basic"
                                                                : "not_binding";
            row["slack"] = slack;
        } else {
            price = pi[i];
            std::vector<double> e(rows, 0.);
            e[i] = 1;
            auto w = b.solver.solve(e);
            double low = -inf, high = inf;
            for (int64_t r = 0; r < rows; ++r) {
                if (std::abs(w[r]) < 1e-11)
                    continue;
                auto kb = b.basic[r];
                double up = (b.hi[kb] - b.value[kb]) / w[r], down = (b.lo[kb] - b.value[kb]) / w[r];
                if (w[r] > 0)
                    high = std::min(high, up), low = std::max(low, down);
                else
                    high = std::min(high, down), low = std::max(low, up);
            }
            if (b.state[k] == State::Upper && std::isfinite(lp.rl[i]))
                low = std::max(low, lp.rl[i] - lp.ru[i]);
            if (b.state[k] == State::Lower && std::isfinite(lp.ru[i]))
                high = std::min(high, lp.ru[i] - lp.rl[i]);
            json range = json::array({number(act + low), number(act + high)});
            bool upper = b.state[k] == State::Upper || b.state[k] == State::Fixed;
            bool lower = b.state[k] == State::Lower || b.state[k] == State::Fixed;
            if (upper)
                upper_range = range;
            if (lower)
                lower_range = range;
            if (b.state[k] == State::Upper && std::isfinite(lp.rl[i]))
                lower_range = json::array({json(), act});
            if (b.state[k] == State::Lower && std::isfinite(lp.ru[i]))
                upper_range = json::array({act, json()});
            row["status"] = b.state[k] == State::Fixed ? "binding_equality"
                            : upper                    ? "binding_upper"
                                                       : "binding_lower";
            row["slack"] = 0.;
            if (std::abs(price) > dtol)
                binding.push_back({i, sense * price, act + low, act + high, act, upper,
                                   b.state[k] == State::Fixed});
        }
        row["shadow_price"] = price == 0 ? 0. : sense * price;
        row["lower_bound_range"] = lower_range;
        row["upper_bound_range"] = upper_range;
        constraints.push_back(row);
    }

    // Objective-coefficient ranging.
    json variables = json::array();
    bool cost_ranging = rows <= objective_ranging_limit;
    struct Costed {
        int64_t j;
        double reduced;
    };
    std::vector<Costed> priced;
    for (int64_t j = 0; j < n; ++j) {
        double low = -inf, high = inf;
        if (b.state[j] == State::Lower)
            low = -std::max(d[j], 0.);
        else if (b.state[j] == State::Upper)
            high = -std::min(d[j], 0.);
        else if (b.state[j] == State::Free)
            low = high = 0;
        else if (b.state[j] == State::Basic && cost_ranging) {
            std::vector<double> e(rows, 0.);
            e[b.position[j]] = 1;
            auto rho = b.solver.solve(e, true);
            for (int64_t k = 0; k < total; ++k) {
                if (b.state[k] == State::Basic || b.state[k] == State::Fixed)
                    continue;
                double alpha = b.dot(k, rho);
                if (std::abs(alpha) < 1e-11)
                    continue;
                double dk = d[k], ratio = dk / alpha;
                bool lower_side = b.state[k] == State::Lower;
                if (b.state[k] == State::Free)
                    low = std::max(low, 0.), high = std::min(high, 0.);
                else if ((alpha > 0) == lower_side)
                    high = std::min(high, std::max(ratio, 0.));
                else
                    low = std::max(low, std::min(ratio, 0.));
            }
        }
        json v = {{"name", lp.names[j]},
                  {"value", b.value[j]},
                  {"status", state_name(b.state[j])},
                  {"objective_coefficient", sense * lp.c[j]},
                  {"reduced_cost", d[j] == 0 ? 0. : sense * d[j]}};
        if (b.state[j] != State::Basic || cost_ranging)
            v["objective_coefficient_range"] = original_range(lp.c[j] + low, lp.c[j] + high);
        else
            v["objective_coefficient_range"] = nullptr;
        if (fixed_integers && source.types[j] != VarType::Continuous)
            v["fixed_integer"] = true;
        variables.push_back(v);
        if (b.state[j] != State::Basic && std::abs(d[j]) > dtol)
            priced.push_back({j, d[j]});
    }

    // Plain-language findings ranked by economic impact.
    json findings = json::array();
    std::sort(binding.begin(), binding.end(),
              [](auto &a, auto &c) { return std::abs(a.price) > std::abs(c.price); });
    for (size_t t = 0; t < binding.size() && t < 8; ++t) {
        auto &bd = binding[t];
        std::ostringstream s;
        s << "Constraint '" << lp.row_names[bd.row] << "' is "
          << (bd.equality ? "fixed at " : bd.upper ? "binding at its upper limit "
                                                   : "binding at its lower limit ")
          << text(bd.bound) << ". Each unit increase of that limit changes the objective by " << text(bd.price)
          << " (" << (improves(bd.price) ? "improvement" : "deterioration")
          << "), valid while the limit stays within [" << text(bd.low) << ", "
          << text(bd.high) << "].";
        findings.push_back(s.str());
    }
    std::sort(priced.begin(), priced.end(),
              [](auto &a, auto &c) { return std::abs(a.reduced) > std::abs(c.reduced); });
    for (size_t t = 0; t < priced.size() && t < 5; ++t) {
        auto j = priced[t].j;
        std::ostringstream s;
        if (b.state[j] == State::Fixed)
            s << "Variable '" << lp.names[j] << "' is fixed at " << text(b.value[j])
              << "; each unit of that commitment changes the objective by "
              << text(sense * priced[t].reduced) << " relative to the remaining plan.";
        else
            s << "Variable '" << lp.names[j] << "' is held at its "
              << (b.state[j] == State::Upper ? "upper" : "lower") << " bound "
              << text(b.value[j]) << "; its objective coefficient must move by at least "
              << text(std::abs(priced[t].reduced))
              << " before changing its level could improve the plan.";
        findings.push_back(s.str());
    }
    if (findings.empty())
        findings.push_back("No constraint or bound has a nonzero marginal value at this optimum.");

    double objective = lp.offset;
    for (int64_t j = 0; j < n; ++j)
        objective += lp.c[j] * b.value[j];
    auto &st = b.solver.stats;
    json report = {
        {"analysis", "LP_SENSITIVITY"},
        {"status", dual_feasible && primal_violation <= 1e-6 ? "OPTIMAL_BASIS" : "PARTIAL"},
        {"model", lp.name},
        {"fingerprint", source.fingerprint()},
        {"objective", sense * objective},
        {"fixed_integer_analysis", fixed_integers},
        {"basis",
         {{"rows", rows},
          {"structural_basic",
           std::count_if(b.basic.begin(), b.basic.end(), [&](int64_t k) { return k < n; })},
          {"degenerate_pivots", pivot_count},
          {"bound_flips", flips},
          {"dual_feasible", dual_feasible},
          {"pivot_limit_reached", pivot_limited},
          {"primal_violation", primal_violation}}},
        {"linear_algebra",
         {{"factorization", st.fp32_factor ? "FP32 LU + FP64 iterative refinement"
                                           : "FP64 LU"},
          {"solves", st.solves},
          {"refinement_steps", st.refinement_steps},
          {"fp64_fallbacks", st.fp64_fallbacks},
          {"worst_relative_residual", st.worst_relative_residual}}},
        {"verification",
         {{"primal_residual", accuracy.primal},
          {"dual_residual", accuracy.dual},
          {"kkt_error", accuracy.kkt},
          {"relative_gap", accuracy.gap},
          {"safe_dual_bound", number(sense * accuracy.lower_bound)},
          {"passed", accuracy.finite && accuracy.kkt <= 1e-6}}},
        {"constraints", constraints},
        {"variables", variables},
        {"findings", findings},
        {"seconds", elapsed(start)},
        {"scope",
         std::string("Ranges hold one parameter change at a time for this optimal basis; a "
                     "degenerate optimum can admit other bases with different ranges.") +
             (fixed_integers ? " Integer variables are fixed at the incumbent." : "") +
             (cost_ranging ? "" : " Basic-variable cost ranging skipped above 1500 rows.")}};
    return report.dump(2);
}
// LP differentiation at an optimal basis (implicit differentiation of the basic solution;
// Amos & Kolter 2017 for the KKT view). With x_B = -B^{-1} N x_N and w = B^{-T} g_B for an
// upstream gradient g = dl/dx:
//   dl/d(active bound of row i)        = w_i            (basic rows: 0)
//   dl/d(bound of nonbasic column j)   = g_j - w'a_j
//   dl/dA_ij                           = -w_i x_j
// x* is piecewise constant in c, so dl/dc is estimated with perturbed optimizers
// (Berthet et al. 2020): grad ~ E[(g'x*(c+sZ) - g'x*(c-sZ)) Z] / (2s), antithetic samples.
std::string differentiate_json(const Model &source, const Options &options,
                               const std::vector<double> &upstream, int samples, double sigma,
                               uint64_t seed) {
    auto start = Clock::now();
    if (source.is_qp() || source.is_mip())
        return json({{"analysis", "LP_DIFFERENTIATION"}, {"status", "UNSUPPORTED"},
                     {"message", "Differentiation supports continuous LPs"}})
            .dump(2);
    if (!upstream.empty() && upstream.size() != source.c.size())
        return json({{"analysis", "LP_DIFFERENTIATION"}, {"status", "INVALID_INPUT"},
                     {"message", "Upstream gradient length must equal the number of variables"}})
            .dump(2);
    Optimum optimum;
    auto problem = optimal_basis(source, options, nullptr, optimum);
    if (!problem.empty()) {
        auto failed = json::parse(problem);
        failed["analysis"] = "LP_DIFFERENTIATION";
        return failed.dump(2);
    }
    const Model &lp = optimum.lp;
    Basis &b = *optimum.basis;
    const int64_t n = b.n, rows = b.rows, total = n + rows;
    const double sense = lp.sense;
    const auto &pi = optimum.pi;
    std::vector<double> x(b.value.begin(), b.value.begin() + n);
    bool degenerate = false;
    for (int64_t r = 0; r < rows; ++r) {
        auto k = b.basic[r];
        double v = b.value[k];
        if ((std::isfinite(b.lo[k]) && std::abs(v - b.lo[k]) <= 1e-9 * (1 + std::abs(v))) ||
            (std::isfinite(b.hi[k]) && std::abs(v - b.hi[k]) <= 1e-9 * (1 + std::abs(v))))
            degenerate = true;
    }
    auto side_of = [&](int64_t k) -> std::string {
        switch (b.state[k]) {
        case State::Lower: return "lower";
        case State::Upper: return "upper";
        case State::Fixed: return "both";
        case State::Basic: return "none";
        default: return "free";
        }
    };
    // Objective gradients (original sense).
    json obj_rows_lower = json::array(), obj_rows_upper = json::array();
    for (int64_t i = 0; i < rows; ++i) {
        auto st = b.state[n + i];
        double g = st == State::Basic ? 0. : sense * pi[i];
        bool lower = st == State::Lower || st == State::Fixed;
        obj_rows_lower.push_back(lower ? g : 0.);
        obj_rows_upper.push_back(st == State::Upper ? g : 0.);
    }
    json obj_var_lower = json::array(), obj_var_upper = json::array(), obj_c = json::array();
    for (int64_t j = 0; j < n; ++j) {
        double d = b.state[j] == State::Basic ? 0. : sense * b.reduced(j, pi);
        bool lower = b.state[j] == State::Lower || b.state[j] == State::Fixed;
        obj_var_lower.push_back(lower ? d : 0.);
        obj_var_upper.push_back(b.state[j] == State::Upper ? d : 0.);
        obj_c.push_back(x[j]);
    }
    json obj_a = json::array();
    for (int64_t i = 0; i < rows; ++i)
        for (auto k = lp.A.ptr[i]; k < lp.A.ptr[i + 1]; ++k) {
            double v = -sense * pi[i] * x[lp.A.index[k]];
            if (v != 0)
                obj_a.push_back({i, lp.A.index[k], v});
        }
    double objective = lp.offset;
    for (int64_t j = 0; j < n; ++j)
        objective += lp.c[j] * x[j];
    json report = {{"analysis", "LP_DIFFERENTIATION"},
                   {"status", "OPTIMAL_BASIS"},
                   {"model", lp.name},
                   {"objective", sense * objective},
                   {"x", x},
                   {"variable_names", lp.names},
                   {"row_names", lp.row_names},
                   {"degenerate_basis", degenerate},
                   {"objective_gradient",
                    {{"c", obj_c},
                     {"row_lower", obj_rows_lower},
                     {"row_upper", obj_rows_upper},
                     {"variable_lower", obj_var_lower},
                     {"variable_upper", obj_var_upper},
                     {"A", obj_a}}}};
    json status = json::array();
    for (int64_t k = 0; k < total; ++k)
        status.push_back(side_of(k));
    report["active_sides"] = status;
    if (!upstream.empty()) {
        std::vector<double> gb(rows, 0.);
        for (int64_t r = 0; r < rows; ++r)
            if (b.basic[r] < n)
                gb[r] = upstream[b.basic[r]];
        auto w = b.solver.solve(gb, true);
        json row_lower = json::array(), row_upper = json::array();
        for (int64_t i = 0; i < rows; ++i) {
            auto st = b.state[n + i];
            double g = st == State::Basic ? 0. : w[i];
            row_lower.push_back(st == State::Lower || st == State::Fixed ? g : 0.);
            row_upper.push_back(st == State::Upper ? g : 0.);
        }
        json var_lower = json::array(), var_upper = json::array();
        for (int64_t j = 0; j < n; ++j) {
            double g = b.state[j] == State::Basic ? 0. : upstream[j] - b.dot(j, w);
            var_lower.push_back(b.state[j] == State::Lower || b.state[j] == State::Fixed ? g : 0.);
            var_upper.push_back(b.state[j] == State::Upper ? g : 0.);
        }
        json a = json::array();
        for (int64_t i = 0; i < rows; ++i)
            for (auto k = lp.A.ptr[i]; k < lp.A.ptr[i + 1]; ++k) {
                double v = -w[i] * x[lp.A.index[k]];
                if (v != 0)
                    a.push_back({i, lp.A.index[k], v});
            }
        // Perturbed-optimizer estimate of dl/dc.
        std::vector<double> gc(n, 0.);
        int used = 0;
        double scale = 1;
        for (double v : lp.c)
            scale = std::max(scale, std::abs(v));
        double step = sigma * scale;
        if (samples > 0) {
            std::mt19937_64 rng(seed);
            std::normal_distribution<double> normal;
            Options o = options;
            o.device = "cpu";
            o.method = lp.A.rows <= simplex_row_limit() ? "simplex" : "auto";
            auto loss = [&](const std::vector<double> &z, double sign) {
                Model perturbed = lp;
                for (int64_t j = 0; j < n; ++j)
                    perturbed.c[j] += sign * step * z[j];
                auto r = solve(perturbed, o);
                if (r.status != "OPTIMAL")
                    return std::numeric_limits<double>::quiet_NaN();
                long double v = 0;
                for (int64_t j = 0; j < n; ++j)
                    v += (long double)upstream[j] * r.x[j];
                return double(v);
            };
            for (int t = 0; t < samples; ++t) {
                std::vector<double> z(n);
                for (auto &v : z)
                    v = normal(rng);
                double plus = loss(z, 1), minus = loss(z, -1);
                if (!std::isfinite(plus) || !std::isfinite(minus))
                    continue;
                for (int64_t j = 0; j < n; ++j)
                    gc[j] += (plus - minus) * z[j] / (2 * step);
                ++used;
            }
            for (auto &v : gc)
                v = used ? sense * v / used : 0.;
        }
        report["vjp"] = {{"row_lower", row_lower},
                         {"row_upper", row_upper},
                         {"variable_lower", var_lower},
                         {"variable_upper", var_upper},
                         {"A", a},
                         {"c", gc},
                         {"c_method", samples > 0 ? "perturbed optimizer (antithetic Gaussian)"
                                                  : "exact: zero almost everywhere"},
                         {"c_samples_used", used},
                         {"c_perturbation", step}};
    }
    report["seconds"] = elapsed(start);
    report["scope"] = "Exact derivatives of the basic solution for bound and matrix "
                      "parameters; equality rows report d/d(rhs) under row_lower. A degenerate "
                      "basis makes these one-sided directional derivatives. Cost gradients of "
                      "x* are zero almost everywhere; the optional perturbed estimate smooths "
                      "them.";
    return report.dump(2);
}
} // namespace vantage
