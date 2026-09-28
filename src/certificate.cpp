#include "json.hpp"
#include "sha256.hpp"
#include "vantage/vantage.hpp"
#include <stdexcept>
namespace vantage {
namespace {
using J = nlohmann::json;
J number(double x) {
    return std::isfinite(x) ? J(x == 0 ? 0. : x) : J(x < 0 ? "-infinity" : "infinity");
}
J canonical(const Model &m) {
    J v = J::object(), r = J::object(), q = J::object();
    for (size_t i = 0; i < m.c.size(); ++i) {
        v[m.names[i]] = {number(m.lb[i]), number(m.ub[i]), number(m.c[i]), int(m.types[i])};
        std::map<std::string, double> terms;
        terms[m.names[i]] = m.q[i];
        if (!m.Q.value.empty())
            for (auto k = m.Q.ptr[i]; k < m.Q.ptr[i + 1]; ++k)
                terms[m.names[m.Q.index[k]]] += m.Q.value[k];
        q[m.names[i]] = J::object();
        for (auto &[name, value] : terms)
            if (value != 0)
                q[m.names[i]][name] = number(value);
    }
    for (size_t i = 0; i < m.rl.size(); ++i) {
        J terms = J::object();
        std::map<std::string, double> values;
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
            values[m.names[m.A.index[k]]] += m.A.value[k];
        for (auto &[name, value] : values)
            if (value != 0)
                terms[name] = number(value);
        r[m.row_names[i]] = {number(m.rl[i]), number(m.ru[i]), terms};
    }
    return {{"variables", v},
            {"rows", r},
            {"quadratic", q},
            {"offset", number(m.offset)},
            {"sense", m.sense}};
}
double finite(const J &j) {
    if (!j.is_number())
        throw std::runtime_error("Expected finite number");
    double v = j.get<double>();
    if (!std::isfinite(v))
        throw std::runtime_error("Nonfinite number");
    return v;
}
std::vector<double> vector(const J &j, const std::vector<std::string> &names) {
    if (!j.is_object() || j.size() != names.size())
        throw std::runtime_error("Candidate dimensions mismatch");
    std::vector<double> v;
    v.reserve(names.size());
    for (auto &name : names)
        v.push_back(finite(j.at(name)));
    return v;
}
J named(const std::vector<double> &v, const std::vector<std::string> &names) {
    J j = J::object();
    if (v.size() != names.size())
        return j;
    for (size_t i = 0; i < v.size(); ++i)
        j[names[i]] = v[i];
    return j;
}
J replay(const Model &m, const J &c) {
    J checks = J::object();
    bool valid = true;
    auto check = [&](const std::string &name, bool pass, J residual,
                     const std::string &explanation) {
        checks[name] = {{"pass", pass}, {"residual", residual}, {"explanation", explanation}};
        valid &= pass;
    };
    check("model_fingerprint", c.at("model_fingerprint") == certificate_fingerprint(m), nullptr,
          "The original model matches the certificate's SHA-256 fingerprint.");
    double tol = finite(c.at("tolerances").at("feasibility")),
           it = finite(c.at("tolerances").at("integrality")),
           gap = finite(c.at("tolerances").at("gap"));
    if (tol <= 0 || tol > 1e-6 || it <= 0 || it > 1e-6 || gap <= 0 || gap > 1e-4)
        throw std::runtime_error("Tolerance exceeds verifier policy");
    if (c.at("dimensions") != J({{"variables", m.c.size()},
                                 {"constraints", m.rl.size()},
                                 {"nonzeros", m.A.value.size()}}))
        throw std::runtime_error("Model dimensions mismatch");
    if (c.at("objective_sense") != (m.sense == 1 ? "min" : "max"))
        throw std::runtime_error("Objective sense mismatch");
    if (c.at("problem_type") !=
        (m.is_mip() ? (m.is_qp() ? "MIQP" : "MILP") : (m.is_qp() ? "QP" : "LP")))
        throw std::runtime_error("Problem type mismatch");
    std::string status = "UNVERIFIED";
    J metrics = J::object();
    Verifier verifier(m);
    if (c.at("kind") == "infeasibility") {
        if (m.is_qp())
            throw std::runtime_error("Only LP Farkas certificates supported");
        double margin = verifier.infeasibility_bound(vector(c.at("ray"), m.row_names));
        check("farkas", std::isfinite(margin) && margin > 0, number(margin),
              "The outward-rounded Farkas separation margin is strictly positive.");
        status = "INFEASIBLE";
    } else if (c.at("kind") == "solution") {
        auto x = vector(c.at("primal"), m.names), y = vector(c.at("dual"), m.row_names);
        if (m.Q.value.empty() &&
            std::any_of(m.q.begin(), m.q.end(), [](double v) { return v < 0; }))
            throw std::runtime_error("Nonconvex quadratic certificate unsupported");
        auto a = verifier.evaluate(x, y);
        check("primal_feasibility", a.finite && a.primal <= tol, number(a.primal),
              "Every variable bound and original constraint satisfies the configured normalized "
              "tolerance.");
        check("integrality", a.integrality <= it, number(a.integrality),
              "Integer variables are within the configured distance of an integer.");
        double error = std::abs(finite(c.at("objective")) - m.sense * a.objective);
        check("objective_consistency", a.finite && error <= tol * (1 + std::abs(a.objective)),
              number(error), "The objective was recomputed from the original model and candidate.");
        metrics = {{"primal_absolute", number(a.primal_absolute)},
                   {"stationarity", number(a.dual)},
                   {"stationarity_absolute", number(a.dual_absolute)},
                   {"complementarity", number(a.complementarity)},
                   {"kkt", number(a.kkt)},
                   {"relative_kkt_gap", number(a.gap)},
                   {"canonical_lower_bound", number(a.lower_bound)}};
        bool convex = std::all_of(m.q.begin(), m.q.end(), [](double v) { return v >= 0; }) ||
                      !m.Q.value.empty();
        bool optimal = convex && a.finite && a.kkt <= tol;
        auto diagnostic = [&](const std::string &name, bool pass, double residual,
                              const std::string &explanation) {
            checks[name] = {{"pass", pass},
                            {"residual", number(residual)},
                            {"required_for", "optimality"},
                            {"explanation", explanation}};
        };
        diagnostic("stationarity_and_dual_feasibility", a.finite && a.dual <= tol, a.dual,
                   "Projected box stationarity and row-dual sign restrictions were recomputed; "
                   "this check is required for continuous optimality.");
        diagnostic("complementarity", a.finite && a.gap <= tol, a.complementarity,
                   "Complementarity is reported in absolute units; acceptance uses the existing "
                   "normalized complementarity and dual-gap envelope.");
        diagnostic(
            "relaxation_bound", std::isfinite(a.lower_bound), a.lower_bound,
            "An outward-rounded relaxation bound was recomputed over the original variable box.");
        if (m.is_mip()) {
            double ag =
                std::isfinite(a.lower_bound) ? std::max(0., a.objective - a.lower_bound) : inf;
            double rg = ag / std::max(1., std::abs(a.objective));
            metrics["verified_absolute_gap"] = number(ag);
            metrics["verified_relative_gap"] = number(rg);
            diagnostic("mip_gap", std::isfinite(rg) && rg <= gap, rg,
                       "The original-model relaxation bound and feasible integer incumbent close "
                       "the gap only if this check passes.");
            optimal = convex && a.finite && std::isfinite(a.lower_bound) && rg <= gap;
            status =
                optimal ? "OPTIMAL_WITHIN_VERIFIED_GAP_TOLERANCE" : "FEASIBLE_WITH_UNRESOLVED_GAP";
            metrics["tree_bound_verified"] = false;
            if (c.contains("solver_bound_telemetry")) {
                const auto &telemetry = c.at("solver_bound_telemetry");
                double bound = finite(telemetry.at("canonical_global_bound"));
                double claimed_gap = finite(telemetry.at("relative_gap"));
                double expected =
                    std::max(0., a.objective - bound) / std::max(1., std::abs(a.objective));
                check("reported_gap_consistency",
                      bound <= a.objective + tol * (1 + std::abs(a.objective)) &&
                          claimed_gap >= 0 && std::abs(expected - claimed_gap) <= tol,
                      number(std::abs(expected - claimed_gap)),
                      "The reported tree bound and incumbent give the reported gap; this does not "
                      "prove the tree bound.");
                metrics["reported_bound_independently_established"] =
                    std::isfinite(a.lower_bound) && bound <= a.lower_bound;
            }
        } else
            status = optimal ? "OPTIMAL" : "FEASIBLE";
        if (status == "OPTIMAL")
            check("kkt", optimal, number(a.kkt),
                  "Convex original-model projected stationarity, dual signs, complementarity and "
                  "feasibility pass.");
    } else
        throw std::runtime_error("Unknown certificate kind");
    return {{"valid", valid},
            {"status", valid ? status : "INVALID"},
            {"checks", checks},
            {"metrics", metrics}};
}
std::string digest(J c) {
    c.erase("payload_sha256");
    c.erase("timestamp");
    return sha256(c.dump());
}
} // namespace
std::string certificate_fingerprint(const Model &m) {
    m.validate();
    return sha256(canonical(m).dump());
}
std::string certificate_json(const Model &m, const Result &r, const Options &o) {
    J c = {
        {"certificate_schema_version", "1.0"},
        {"solver", "NIRYUKTI"},
        {"solver_version", "0.2.0"},
        {"model_fingerprint", certificate_fingerprint(m)},
        {"dimensions",
         {{"variables", m.c.size()}, {"constraints", m.rl.size()}, {"nonzeros", m.A.value.size()}}},
        {"problem_type", m.is_mip() ? (m.is_qp() ? "MIQP" : "MILP") : (m.is_qp() ? "QP" : "LP")},
        {"objective_sense", m.sense == 1 ? "min" : "max"},
        {"termination_status", r.status},
        {"configuration",
         {{"backend", r.backend},
          {"method", o.method},
          {"scaling", o.scaling},
          {"solver_tolerance", o.tol},
          {"presolve", o.presolve},
          {"scaling_passes", o.scaling_passes},
          {"restart", o.restart},
          {"adaptive", o.adaptive},
          {"branching", o.branching},
          {"node_selection", o.node_selection},
          {"iteration_limit", o.iteration_limit},
          {"node_limit", o.node_limit},
          {"time_limit", o.time_limit},
          {"threads", o.threads},
          {"cuts", o.cuts},
          {"polishing", o.polishing},
          {"matrix_precision", o.matrix_precision},
          {"primal_weight", o.primal_weight}}},
        {"tolerances",
         {{"feasibility", std::min(o.tol, 1e-6)},
          {"integrality", std::min(o.integer_tol, 1e-6)},
          {"gap", std::min(o.mip_gap, 1e-4)}}}};
    if (!r.infeasibility_ray.empty()) {
        c["kind"] = "infeasibility";
        c["ray"] = named(r.infeasibility_ray, m.row_names);
    } else {
        c["kind"] = "solution";
        c["primal"] = named(r.x, m.names);
        c["dual"] = named(r.y, m.row_names);
        c["objective"] = m.sense * r.accuracy.objective;
    }
    if (m.is_mip() && std::isfinite(r.best_bound) && std::isfinite(r.mip_gap))
        c["solver_bound_telemetry"] = {{"canonical_global_bound", r.best_bound},
                                       {"relative_gap", r.mip_gap}};
#ifdef VANTAGE_GIT_COMMIT
    c["git_commit"] = VANTAGE_GIT_COMMIT;
#endif
    try {
        c["verification"] = replay(m, c);
    } catch (const std::exception &e) {
        c["verification"] = {{"valid", false}, {"status", "INVALID"}, {"reason", e.what()}};
    }
    c["verification_evidence"] = {{"tree_optimality_replayed", false}};
    c["payload_sha256"] = digest(c);
    return c.dump(2);
}
std::string verify_certificate_json(const Model &m, const std::string &text) {
    try {
        if (text.size() > 64 * 1024 * 1024)
            throw std::runtime_error("Certificate exceeds 64 MiB limit");
        auto c = J::parse(text, [](int depth, J::parse_event_t, const J &) {
            if (depth > 32)
                throw std::runtime_error("Certificate nesting limit");
            return true;
        });
        if (c.at("certificate_schema_version") != "1.0")
            throw std::runtime_error("Unsupported certificate schema");
        bool payload_matches = c.at("payload_sha256") == digest(c);
        // Diagnose common mathematical edits before returning a generic checksum failure.
        if (!payload_matches && c.contains("kind") && c.contains("model_fingerprint") &&
            c.at("model_fingerprint") == certificate_fingerprint(m)) {
            auto replayed = replay(m, c);
            for (auto key : {"objective_consistency", "primal_feasibility", "integrality",
                             "stationarity_and_dual_feasibility", "complementarity",
                             "reported_gap_consistency", "farkas"}) {
                if (replayed.at("checks").contains(key) &&
                    !replayed.at("checks").at(key).at("pass").get<bool>()) {
                    std::string reason =
                        std::string(key) == "objective_consistency" ? "objective mismatch" : key;
                    if (reason != "objective mismatch")
                        std::replace(reason.begin(), reason.end(), '_', ' ');
                    return J({{"valid", false}, {"status", "INVALID"}, {"reason", reason}}).dump(2);
                }
            }
        }
        if (!payload_matches)
            throw std::runtime_error("Certificate payload integrity mismatch");
        // Required metadata is structural, never evidence of mathematical validity.
        for (auto key : {"solver", "solver_version", "dimensions", "problem_type",
                         "objective_sense", "termination_status", "configuration"})
            c.at(key);
        for (auto key :
             {"solver", "solver_version", "problem_type", "objective_sense", "termination_status"})
            if (!c.at(key).is_string())
                throw std::runtime_error("Invalid metadata type");
        if (c.at("solver") != "NIRYUKTI" || !c.at("configuration").is_object())
            throw std::runtime_error("Invalid solver metadata");
        auto report = replay(m, c);
        if (report != c.at("verification"))
            throw std::runtime_error("Stored verification claims differ from independent replay");
        return report.dump(2);
    } catch (const std::exception &e) {
        return J({{"valid", false}, {"status", "INVALID"}, {"reason", e.what()}}).dump(2);
    }
}
} // namespace vantage
