#include "farkas.hpp"
#include "internal.hpp"
#include "json.hpp"
#include "vantage/analysis.hpp"
#include <iomanip>
#include <sstream>
// Irreducible infeasible subsystem (IIS) search.
// 1. Seed with the support of a verified Farkas ray of the full model.
// 2. Deletion filter (Chinneck & Dravnieks 1991): drop each member whose removal leaves the
//    subsystem infeasible. Every infeasible verdict requires an independently verified
//    Farkas certificate; every feasible verdict requires a verified primal point.
// 3. For each IIS member, an elastic LP measures the smallest change of that single member
//    that makes the complete model feasible.
namespace vantage {
namespace {
using json = nlohmann::json;
enum class Verdict { Feasible, Infeasible, Unknown };
struct Member {
    bool row;
    int64_t index;
    bool lower;
};
std::string text(double v) {
    std::ostringstream s;
    s << std::setprecision(6) << (std::abs(v) < 1e-12 ? 0. : v);
    return s.str();
}
std::vector<Entry> entries(const Sparse &a) {
    std::vector<Entry> e;
    for (int64_t i = 0; i < a.rows; ++i)
        for (auto k = a.ptr[i]; k < a.ptr[i + 1]; ++k)
            e.push_back({i, a.index[k], a.value[k]});
    return e;
}
class Oracle {
    const Model &m;
    std::vector<Entry> all;
    Options options;

  public:
    int64_t solves = 0;
    std::vector<double> last_ray;
    std::vector<int64_t> last_rows;
    double last_margin = 0;
    double big = 1e6;
    Oracle(const Model &model, const Options &o) : m(model), all(entries(model.A)), options(o) {
        options.device = "cpu";
        options.presolve = true;
        double scale = 1;
        for (auto *v : {&m.lb, &m.ub, &m.rl, &m.ru})
            for (double b : *v)
                if (std::isfinite(b))
                    scale = std::max(scale, std::abs(b));
        big *= scale;
    }
    Verdict test(const std::vector<char> &rows, const std::vector<char> &lower,
                 const std::vector<char> &upper) {
        ++solves;
        Model s;
        s.name = m.name + "_subsystem";
        size_t n = m.c.size();
        s.c.assign(n, 0.);
        s.q.assign(n, 0.);
        s.types.assign(n, VarType::Continuous);
        s.names = m.names;
        s.lb.resize(n);
        s.ub.resize(n);
        for (size_t j = 0; j < n; ++j) {
            s.lb[j] = lower[j] ? m.lb[j] : -inf;
            s.ub[j] = upper[j] ? m.ub[j] : inf;
        }
        std::vector<int64_t> map(m.rl.size(), -1);
        std::vector<int64_t> kept;
        for (size_t i = 0; i < m.rl.size(); ++i)
            if (rows[i]) {
                map[i] = int64_t(kept.size());
                kept.push_back(int64_t(i));
                s.rl.push_back(m.rl[i]);
                s.ru.push_back(m.ru[i]);
                s.row_names.push_back(m.row_names[i]);
            }
        std::vector<Entry> e;
        for (auto &t : all)
            if (map[t.row] >= 0)
                e.push_back({map[t.row], t.col, t.value});
        s.A = Sparse::build(int64_t(kept.size()), int64_t(n), std::move(e));
        // A bound-only subsystem is infeasible only if some lb > ub, excluded by validation.
        if (kept.empty())
            return Verdict::Feasible;
        // Removed bounds leave free columns. A big-box surrogate keeps every LP well posed;
        // verdicts stay rigorous because each is checked against the true subsystem `s`:
        // surrogate-feasible points are truly feasible (smaller box), and rays must pass
        // the verifier with the true infinite bounds.
        Model box = s;
        for (size_t j = 0; j < n; ++j) {
            if (!std::isfinite(box.lb[j]))
                box.lb[j] = -big;
            if (!std::isfinite(box.ub[j]))
                box.ub[j] = big;
        }
        Verifier verifier(s);
        std::vector<double> zero(kept.size(), 0.);
        const std::pair<const Model *, const char *> attempts[] = {
            {&box, "simplex"}, {&s, "auto"}, {&box, "pdhg"}, {&s, "pdhg"}};
        for (auto [model, method] : attempts) {
            Options o = options;
            o.method = method;
            if (o.method == "simplex" && model->A.rows > simplex_row_limit())
                continue;
            auto r = solve(*model, o);
            if (r.status == "INFEASIBLE") {
                std::vector<double> ray;
                double margin = certify_infeasible(s, r, &ray);
                if (margin > 0 && std::isfinite(margin)) {
                    last_ray = ray;
                    last_rows = kept;
                    last_margin = margin;
                    return Verdict::Infeasible;
                }
            }
            if (r.status == "OPTIMAL" && r.x.size() == n) {
                auto a = verifier.evaluate(r.x, zero, false);
                if (a.finite && a.primal <= 1e-7)
                    return Verdict::Feasible;
            }
        }
        return Verdict::Unknown;
    }
};

// Smallest single-member relaxation that makes the full model feasible.
json repair(const Model &m, const Member &member, const Options &base) {
    Model e = m;
    std::fill(e.c.begin(), e.c.end(), 0.);
    std::fill(e.q.begin(), e.q.end(), 0.);
    e.Q = Sparse();
    e.offset = 0;
    e.sense = 1;
    std::fill(e.types.begin(), e.types.end(), VarType::Continuous);
    auto add_var = [&](const std::string &name) {
        e.c.push_back(1);
        e.q.push_back(0);
        e.lb.push_back(0);
        e.ub.push_back(inf);
        e.types.push_back(VarType::Continuous);
        e.names.push_back(name);
        return int64_t(e.c.size() - 1);
    };
    auto list = entries(m.A);
    int64_t rows = m.A.rows;
    int64_t up = -1, down = -1;
    if (member.row) {
        up = add_var("__elastic_up");     // lowers the effective lower limit
        down = add_var("__elastic_down"); // raises the effective upper limit
        list.push_back({member.index, up, 1});
        list.push_back({member.index, down, -1});
    } else {
        auto j = member.index;
        auto slack = add_var("__elastic_bound");
        if (member.lower) {
            e.lb[j] = -inf;
            list.push_back({rows, j, 1});
            list.push_back({rows, slack, 1});
            e.rl.push_back(m.lb[j]);
            e.ru.push_back(inf);
        } else {
            e.ub[j] = inf;
            list.push_back({rows, j, 1});
            list.push_back({rows, slack, -1});
            e.rl.push_back(-inf);
            e.ru.push_back(m.ub[j]);
        }
        e.row_names.push_back("__elastic_row");
        ++rows;
        up = slack;
    }
    e.A = Sparse::build(rows, int64_t(e.c.size()), std::move(list));
    Result r;
    for (const char *method : {"simplex", "auto"}) {
        Options o = base;
        o.device = "cpu";
        o.method = method;
        if (o.method == "simplex" && e.A.rows > simplex_row_limit())
            continue;
        r = solve(e, o);
        if (r.status == "OPTIMAL" || r.status == "INFEASIBLE")
            break;
    }
    if (r.status == "INFEASIBLE")
        return {{"restores_feasibility", false},
                {"reason", "Other conflicts remain even if this member is relaxed"}};
    if (r.status != "OPTIMAL")
        return {{"restores_feasibility", nullptr}, {"reason", "Elastic solve: " + r.status}};
    double amount = r.accuracy.objective;
    json out = {{"restores_feasibility", true}, {"minimum_change", amount}};
    std::ostringstream s;
    if (member.row) {
        const auto &name = m.row_names[member.index];
        bool equality = m.rl[member.index] == m.ru[member.index];
        if (r.x[up] >= r.x[down]) {
            out["direction"] = "decrease_lower_limit";
            out["new_limit"] = m.rl[member.index] - r.x[up];
            s << (equality ? "Lower the required value of '" : "Lower the minimum of '") << name << "' from " << text(m.rl[member.index])
              << " to " << text(m.rl[member.index] - r.x[up]);
        } else {
            out["direction"] = "increase_upper_limit";
            out["new_limit"] = m.ru[member.index] + r.x[down];
            s << (equality ? "Raise the required value of '" : "Raise the maximum of '") << name << "' from " << text(m.ru[member.index])
              << " to " << text(m.ru[member.index] + r.x[down]);
        }
    } else {
        const auto &name = m.names[member.index];
        double bound = member.lower ? m.lb[member.index] : m.ub[member.index];
        double target = member.lower ? bound - amount : bound + amount;
        out["direction"] = member.lower ? "decrease_lower_bound" : "increase_upper_bound";
        out["new_limit"] = target;
        s << (member.lower ? "Lower the lower bound of '" : "Raise the upper bound of '") << name
          << "' from " << text(bound) << " to " << text(target);
    }
    s << " (change " << text(amount) << ") to make the whole model feasible.";
    out["advice"] = s.str();
    return out;
}

std::string describe_row(const Model &m, const Sparse &at_rows, int64_t i) {
    std::ostringstream s;
    int shown = 0;
    for (auto k = at_rows.ptr[i]; k < at_rows.ptr[i + 1]; ++k) {
        if (shown == 8) {
            s << " + ...";
            break;
        }
        double v = at_rows.value[k];
        s << (shown ? (v < 0 ? " - " : " + ") : (v < 0 ? "-" : ""));
        if (std::abs(v) != 1)
            s << text(std::abs(v)) << "*";
        s << m.names[at_rows.index[k]];
        ++shown;
    }
    if (!shown)
        s << "0";
    std::string expr = s.str();
    if (m.rl[i] == m.ru[i])
        return expr + " = " + text(m.rl[i]);
    if (std::isfinite(m.rl[i]) && std::isfinite(m.ru[i]))
        return text(m.rl[i]) + " <= " + expr + " <= " + text(m.ru[i]);
    if (std::isfinite(m.rl[i]))
        return expr + " >= " + text(m.rl[i]);
    return expr + " <= " + text(m.ru[i]);
}
} // namespace

std::string iis_json(const Model &source, const Options &options) {
    auto start = Clock::now();
    Model m = source;
    bool relaxed_integrality = m.is_mip();
    std::fill(m.types.begin(), m.types.end(), VarType::Continuous);
    if (m.is_qp()) {
        std::fill(m.q.begin(), m.q.end(), 0.);
        m.Q = Sparse();
    }
    std::fill(m.c.begin(), m.c.end(), 0.);
    size_t n = m.c.size(), rows = m.rl.size();
    Oracle oracle(m, options);
    std::vector<char> all_rows(rows, 1), lower(n), upper(n);
    for (size_t j = 0; j < n; ++j) {
        lower[j] = std::isfinite(m.lb[j]);
        upper[j] = std::isfinite(m.ub[j]);
    }
    json report = {{"analysis", "IIS"}, {"model", source.name},
                   {"fingerprint", source.fingerprint()},
                   {"integrality_relaxed", relaxed_integrality}};
    auto verdict = oracle.test(all_rows, lower, upper);
    if (verdict != Verdict::Infeasible) {
        report["status"] = verdict == Verdict::Feasible ? "FEASIBLE" : "UNKNOWN";
        report["message"] = verdict == Verdict::Feasible
                                ? (relaxed_integrality
                                       ? "The continuous relaxation is feasible; integer "
                                         "infeasibility is not diagnosed by this tool"
                                       : "The model is feasible; no conflict exists")
                                : "Infeasibility could not be certified";
        report["seconds"] = elapsed(start);
        return report.dump(2);
    }
    // Seed from Farkas support.
    std::vector<char> rset(rows, 0), lset(n, 0), uset(n, 0);
    const auto &at = m.A;
    for (size_t t = 0; t < oracle.last_rows.size(); ++t)
        if (std::abs(oracle.last_ray[t]) > 1e-12) {
            auto i = oracle.last_rows[t];
            rset[i] = 1;
            for (auto k = at.ptr[i]; k < at.ptr[i + 1]; ++k) {
                auto j = at.index[k];
                lset[j] = lower[j];
                uset[j] = upper[j];
            }
        }
    if (oracle.test(rset, lset, uset) != Verdict::Infeasible) {
        rset = all_rows;
        lset = lower;
        uset = upper;
    }
    bool complete = true;
    auto limit_hit = [&]() { return elapsed(start) > options.time_limit; };
    auto filter = [&](std::vector<char> &set, size_t count) {
        for (size_t k = 0; k < count; ++k) {
            if (!set[k])
                continue;
            if (limit_hit()) {
                complete = false;
                return;
            }
            set[k] = 0;
            auto v = oracle.test(rset, lset, uset);
            if (v != Verdict::Infeasible)
                set[k] = 1;
            if (v == Verdict::Unknown)
                complete = false;
        }
    };
    filter(rset, rows);
    filter(lset, n);
    filter(uset, n);
    // Final certificate for the reported subsystem.
    auto final_verdict = oracle.test(rset, lset, uset);
    std::vector<double> multiplier(rows, 0.);
    for (size_t t = 0; t < oracle.last_rows.size(); ++t)
        multiplier[oracle.last_rows[t]] = oracle.last_ray[t];
    std::vector<Member> members;
    for (size_t i = 0; i < rows; ++i)
        if (rset[i])
            members.push_back({true, int64_t(i), false});
    for (size_t j = 0; j < n; ++j) {
        if (lset[j])
            members.push_back({false, int64_t(j), true});
        if (uset[j])
            members.push_back({false, int64_t(j), false});
    }
    json conflict_rows = json::array(), conflict_bounds = json::array(), story = json::array();
    for (auto &mb : members) {
        auto fix = repair(m, mb, options);
        if (mb.row) {
            auto i = mb.index;
            auto expr = describe_row(source, source.A, i);
            conflict_rows.push_back({{"name", source.row_names[i]},
                                     {"expression", expr},
                                     {"farkas_multiplier", multiplier[i]},
                                     {"repair", fix}});
            story.push_back("Constraint '" + source.row_names[i] + "': " + expr);
        } else {
            auto j = mb.index;
            double bound = mb.lower ? source.lb[j] : source.ub[j];
            conflict_bounds.push_back({{"variable", source.names[j]},
                                       {"side", mb.lower ? "lower" : "upper"},
                                       {"bound", bound},
                                       {"repair", fix}});
            story.push_back("Bound: " + source.names[j] + (mb.lower ? " >= " : " <= ") +
                            text(bound));
        }
    }
    std::ostringstream summary;
    summary << "These " << members.size()
            << " requirements cannot all hold together, and removing any single one of them "
               "removes this conflict. Weighting the constraints by the Farkas multipliers and "
               "adding them gives an inequality that no point within the listed bounds can "
               "satisfy (verified margin "
            << text(oracle.last_margin) << " in units of the weighted combination).";
    report["status"] = final_verdict == Verdict::Infeasible && complete ? "IIS_FOUND"
                       : final_verdict == Verdict::Infeasible           ? "INFEASIBLE_SUBSYSTEM"
                                                                        : "UNKNOWN";
    report["irreducible"] = complete && final_verdict == Verdict::Infeasible;
    report["rows"] = conflict_rows;
    report["bounds"] = conflict_bounds;
    report["members"] = story;
    report["explanation"] = summary.str();
    report["certificate"] = {{"kind", "BOX_ROW_FARKAS"},
                             {"verified_margin", oracle.last_margin},
                             {"multipliers_by_row", multiplier}};
    report["oracle_solves"] = oracle.solves;
    report["original_rows"] = rows;
    report["original_variables"] = n;
    report["seconds"] = elapsed(start);
    report["scope"] = std::string("Each infeasible verdict carries an independently verified "
                                  "Farkas certificate; feasible verdicts carry verified points. "
                                  "Repairs relax one member against the complete model.") +
                      (relaxed_integrality ? " Integrality was relaxed." : "");
    return report.dump(2);
}
} // namespace vantage
