#include "json.hpp"
#include "vantage/vantage.hpp"
#include "vantage/analysis.hpp"
#include "session.hpp"
#include "../src/gnn.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace vantage;
namespace {
nlohmann::json analyze_model(const Model &m, const Options &o) {
    validate_options(o);
    auto hardware = hardware_info();
    auto advice = advise_model(m, o, hardware);
    int64_t binary = 0, integer = 0, bounded = 0, fixed = 0;
    for (size_t j = 0; j < m.c.size(); ++j) {
        binary += m.types[j] == VarType::Binary;
        integer += m.types[j] != VarType::Continuous;
        bounded += std::isfinite(m.lb[j]) && std::isfinite(m.ub[j]);
        fixed += m.lb[j] == m.ub[j];
    }
    int64_t equalities = 0, ranged = 0;
    for (size_t i = 0; i < m.rl.size(); ++i) {
        equalities += m.rl[i] == m.ru[i];
        ranged += std::isfinite(m.rl[i]) && std::isfinite(m.ru[i]) && m.rl[i] != m.ru[i];
    }
    double min_a = inf, max_a = 0;
    for (double a : m.A.value) if (a != 0) {
        min_a = std::min(min_a, std::abs(a));
        max_a = std::max(max_a, std::abs(a));
    }
    int64_t q_diagonal = 0, q_off_diagonal = 0;
    auto diagonal = m.q;
    for (size_t i = 0; i < m.Q.rows; ++i)
        for (auto k = m.Q.ptr[i]; k < m.Q.ptr[i + 1]; ++k)
            if (m.Q.index[k] == int64_t(i)) diagonal[i] += m.Q.value[k];
            else if (m.Q.index[k] > int64_t(i)) q_off_diagonal += m.Q.value[k] != 0;
    for (double q : diagonal) q_diagonal += q != 0;
    // Connected components of the variable incidence graph describe block structure;
    // this analysis does not reorder or decompose the optimization problem.
    std::vector<int64_t> parent(m.c.size());
    for (size_t j = 0; j < parent.size(); ++j) parent[j] = int64_t(j);
    auto root = [&](int64_t j) { while (parent[j] != j) { parent[j] = parent[parent[j]]; j = parent[j]; } return j; };
    auto join = [&](int64_t a, int64_t b) { a=root(a); b=root(b); if (a!=b) parent[b]=a; };
    for (int64_t i = 0; i < m.A.rows; ++i)
        for (auto k = m.A.ptr[i] + 1; k < m.A.ptr[i + 1]; ++k)
            join(m.A.index[m.A.ptr[i]],m.A.index[k]);
    for (size_t i = 0; i < m.Q.rows; ++i)
        for (auto k = m.Q.ptr[i]; k < m.Q.ptr[i + 1]; ++k)
            if (m.Q.index[k] > int64_t(i)) join(int64_t(i),m.Q.index[k]);
    int64_t components = 0;
    for (size_t j = 0; j < parent.size(); ++j) components += root(int64_t(j)) == int64_t(j);
    long double cells = static_cast<long double>(m.A.rows) * m.A.cols;
    double density = cells > 0 ? double(m.A.value.size() / cells) : 0;
    long double csr_bytes = static_cast<long double>(m.A.value.size()) * (sizeof(double)+sizeof(int64_t)) +
                            static_cast<long double>(m.A.rows+1) * sizeof(int64_t);
    long double dense_bytes = cells * sizeof(double);
    bool csr_smaller = dense_bytes == 0 || csr_bytes < dense_bytes;
    int64_t continuous = int64_t(m.c.size()) - integer;
    std::string type = m.is_mip() ? (m.is_qp() ? "MIQP" : "MILP") : (m.is_qp() ? "QP" : "LP");
    std::string algorithm = advice.method;
    if (m.is_mip()) algorithm = std::string(o.cuts ? "branch-and-cut with " : "branch-and-bound with ") + advice.method + " relaxations";
    return {{"mode", (o.method == "auto" || o.device == "auto") ? "auto" : "explicit"},
        {"problem_class", type},
        {"model", {{"variables",m.c.size()},{"constraints",m.A.rows},{"nonzeros",m.A.value.size()},
                   {"integer_variables",integer},{"binary_variables",binary},{"continuous_variables",continuous},
                   {"bounded_variables",bounded},{"fixed_variables",fixed},{"equality_constraints",equalities},
                   {"ranged_constraints",ranged},{"quadratic_diagonal_terms",q_diagonal},
                   {"quadratic_off_diagonal_terms",q_off_diagonal},{"matrix_density",density},
                   {"matrix_coefficients",{{"minimum_absolute",std::isfinite(min_a)?nlohmann::json(min_a):nlohmann::json()},
                                             {"maximum_absolute",max_a},
                                             {"range",std::isfinite(min_a)&&min_a>0?max_a/min_a:1.0}}},
                   {"incidence_components",components},{"incidence_components_are_solver_decomposition",false},
                   {"csr_uses_less_estimated_storage_than_dense",csr_smaller}}},
        {"hardware",{{"cpu","available"},{"cuda_available",hardware.gpu_available},
                      {"cuda_description",cuda_description()},
                      {"free_gpu_bytes",hardware.free_gpu_bytes},{"total_gpu_bytes",hardware.total_gpu_bytes}}},
        {"selection",{{"device",advice.device},{"method",advice.method},{"algorithm",algorithm},
                       {"reason",advice.reason},{"estimated_gpu_bytes",advice.estimated_gpu_bytes},
                       {"profile","none"}}},
        {"configuration",{{"scaling",o.scaling},{"scaling_passes",o.scaling_passes},
                           {"branching",o.branching},{"node_selection",o.node_selection},
                           {"cuts_enabled",o.cuts},{"primal_heuristic",o.primal_heuristic},
                           {"precision",o.matrix_precision},{"tolerance",o.tol},
                           {"time_limit_seconds",o.time_limit},{"iteration_limit",o.iteration_limit},
                           {"node_limit",o.node_limit},{"presolve",o.presolve}}},
        {"reasons",nlohmann::json::array({advice.reason})},
        {"runtime_estimate",nullptr},
        {"verification_path",m.is_mip()?"original-model incumbent feasibility, integrality, objective, and available replayable relaxation bound"
                                      :"original-model feasibility, objective consistency, stationarity and KKT checks"}};
}
// Three-state assurance badge derived only from independently verified quantities.
nlohmann::json assurance(const Model &m, const Result &r, const Options &o) {
    const auto &a = r.accuracy;
    auto value = [](double v) { return std::isfinite(v) ? nlohmann::json(v) : nlohmann::json(); };
    if (r.status == "INFEASIBLE" && !r.infeasibility_ray.empty() && r.certificate_margin > 0)
        return {{"state","CERTIFIED_INFEASIBLE"},{"color","green"},
                {"meaning","Verified Farkas certificate proves no feasible plan exists"}};
    bool feasible = a.finite && a.primal <= 1e-6 && a.integrality <= 1e-6;
    if (!feasible)
        return {{"state","UNVERIFIED"},{"color","red"},
                {"meaning","No independently verified feasible plan"}};
    double incumbent = a.objective;
    double bound = m.is_mip() ? r.best_bound : a.lower_bound;
    double gap = std::isfinite(bound) ? std::max(0., incumbent - bound) : inf;
    double allowed = std::max(1e-9, (m.is_mip() ? o.mip_gap : o.tol) * std::max(1., std::abs(incumbent)));
    nlohmann::json badge = {{"objective",m.sense*incumbent},
                            {m.sense > 0 ? "lower_bound" : "upper_bound", std::isfinite(bound) ? value(m.sense*bound) : nlohmann::json()},
                            {"unverified_gap", value(gap)}};
    if (!m.is_mip() && r.status == "OPTIMAL" && a.kkt <= o.tol && gap <= allowed) {
        badge["state"] = "CERTIFIED_OPTIMAL"; badge["color"] = "green";
        badge["meaning"] = "Independent KKT checks pass and the verified dual bound meets the plan";
    } else if (std::isfinite(gap)) {
        badge["state"] = "BOUNDED"; badge["color"] = "amber";
        badge["meaning"] = m.is_mip()
            ? "Verified incumbent; the tree bound limits how much better any plan can be (tree not replayed)"
            : "Verified feasible plan with a verified dual bound";
    } else {
        badge["state"] = "FEASIBLE_ONLY"; badge["color"] = "red";
        badge["meaning"] = "Verified feasible plan without a valid bound";
    }
    return badge;
}
}
int main(int argc, char **argv) {
    std::signal(SIGINT, [](int) { interrupted = 1; });
    try {
        if (argc < 2) {
            std::cout
                << "NIRYUKTI 0.2 — Independent Sparse Optimization Engine\nCommands: solve MODEL, analyze MODEL, inspect MODEL, explain MODEL, session MODEL, "
                   "verify MODEL SOLUTION, convert INPUT OUTPUT, devices\n"
                   "          sensitivity MODEL [SOLUTION], iis MODEL, global BILINEAR.json,\n"
                   "          recommend MODEL --history FILE, decompose MODEL [--linking ROWS],\n"
                   "          train-branching OUT.json MODEL... [--dives N --epochs N],\n"
                   "          differentiate MODEL [--upstream grad.json --samples K --sigma S]\nSolve: --device "
                   "cpu|cuda|auto --tol 1e-6 --time-limit 60 --iterations 100000\n       "
                   "--json-out result.json --warm-start result.json --threads 1 --verbose\n  "
                   "     --no-presolve --scaling-passes 5 --no-restart --no-adaptive\n       "
                   "--node-limit 10000 --mip-gap 1e-4 --check-every 100\n"
                   "       --scaling ruiz|combined --method "
                   "auto|simplex|dual-simplex|barrier|concurrent|pdhg|halpern|rhpdhg|r2hpdhg\n"
                   "       --primal-weight displacement|pid --power-iterations 0 --polishing\n"
                   "       --cuda-graphs --gpu-indices auto|32|64 --matrix-precision fp64|mixed\n"
                   "       --newton-precision fp64|mixed (barrier FP32 factor + FP64 refinement)\n"
                   "       --gpu-monitor --certificate-out certificate.json\n"
                   "       --node-selection best-bound|depth-first|best-estimate\n"
                   "       --branching fractional|reliability|gnn [--branching-model w.json] --cuts\n       --primal-heuristic "
                   "repair|pump|rins|local|all\n"
                   "       --checkpoint-out state.json --checkpoint-nodes 100 --resume state.json\n"
                   "       --gpu-presolve --batch-strong-branching\n"
                   "       --auto (alias for --method auto) --dry-run\n"
                   "       --history FILE (record outcome) --method learned (needs --history)\n"
                   "       halpern: experimental CPU/CUDA LP, requires --no-adaptive\n";
            return 0;
        }
        std::string cmd = argv[1];
        if (cmd == "devices") {
            std::cout << "CPU: available\nCUDA: " << cuda_description() << '\n';
            return 0;
        }
        if (argc < 3)
            throw std::runtime_error("Model path required");
        if (cmd == "session") return run_session(argv[2]);
        // Options shared by the analysis commands.
        auto analysis_options = [&](int first, Options &o, std::string *history) {
            for (int i = first; i < argc; ++i) {
                std::string flag = argv[i];
                if (i + 1 >= argc) throw std::runtime_error("Missing option value for " + flag);
                std::string value = argv[++i];
                if (flag == "--time-limit") o.time_limit = std::stod(value);
                else if (flag == "--node-limit") o.node_limit = std::stoll(value);
                else if (flag == "--mip-gap") o.mip_gap = std::stod(value);
                else if (flag == "--tol") o.tol = std::stod(value);
                else if (flag == "--method") o.method = value;
                else if (flag == "--matrix-precision") o.matrix_precision = value;
                else if (flag == "--history" && history) *history = value;
                else throw std::runtime_error("Unsupported option for " + cmd + ": " + flag);
            }
        };
        if (cmd == "train-branching") {
            // niryukti train-branching OUTPUT.json MODEL... [--dives N --epochs N --seed N]
            Options o;
            o.time_limit = 600;
            std::vector<std::string> paths;
            int dives = 6, epochs = 60;
            uint64_t seed = 1;
            for (int i = 3; i < argc; ++i) {
                std::string flag = argv[i];
                if (flag.rfind("--", 0) != 0) { paths.push_back(flag); continue; }
                if (i + 1 >= argc) throw std::runtime_error("Missing option value for " + flag);
                std::string value = argv[++i];
                if (flag == "--dives") dives = std::stoi(value);
                else if (flag == "--epochs") epochs = std::stoi(value);
                else if (flag == "--seed") seed = std::stoull(value);
                else if (flag == "--time-limit") o.time_limit = std::stod(value);
                else throw std::runtime_error("Unsupported option for train-branching: " + flag);
            }
            if (paths.empty() || dives < 1 || epochs < 1)
                throw std::runtime_error("train-branching OUTPUT MODEL... [--dives N --epochs N]");
            std::cout << nlohmann::json::parse(train_branching_json(paths, argv[2], o, dives, epochs, seed)).dump(2) << '\n';
            return 0;
        }
        if (cmd == "global") {
            Options o;
            o.node_limit = 100000;
            analysis_options(3, o, nullptr);
            auto report = nlohmann::json::parse(global_solve_json(argv[2], o));
            std::cout << report.dump(2) << '\n';
            auto status = report.at("status").get<std::string>();
            return status == "GLOBAL_OPTIMAL" || status == "INFEASIBLE" ? 0 : 2;
        }
        auto parse_start = std::chrono::steady_clock::now();
        auto m = read_model(argv[2]);
        double parse_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - parse_start).count();
        if (cmd == "inspect" || cmd == "explain" || cmd == "analyze") {
            nlohmann::json j = {
                {"name", m.name},
                {"type", m.is_mip()  ? (m.is_qp() ? "MIQP" : "MILP")
                         : m.is_qp() ? "QP"
                                     : "LP"},
                {"rows", m.A.rows},
                {"columns", m.A.cols},
                {"nonzeros", m.A.value.size()},
                {"fingerprint", m.fingerprint()},
                {"gpu_storage_estimate_bytes",
                 48. * (m.A.value.size() + m.Q.value.size()) + 240. * (m.A.rows + m.A.cols + 2)}};
            Options automatic; automatic.method = "auto";
            automatic.device = "auto";
            if (cmd == "analyze") {
                for (int i = 3; i < argc; ++i) {
                    std::string flag = argv[i];
                    if (flag == "--auto") { automatic.method = "auto"; continue; }
                    if (flag == "--cuts") { automatic.cuts = true; continue; }
                    if (flag == "--no-presolve") { automatic.presolve = false; continue; }
                    if (i + 1 >= argc) throw std::runtime_error("Missing option value for " + flag);
                    std::string value = argv[++i];
                    if (flag == "--device") automatic.device = value;
                    else if (flag == "--method") automatic.method = value;
                    else if (flag == "--scaling") automatic.scaling = value;
                    else if (flag == "--branching") automatic.branching = value;
                    else if (flag == "--node-selection") automatic.node_selection = value;
                    else if (flag == "--primal-heuristic") automatic.primal_heuristic = value;
                    else if (flag == "--matrix-precision") automatic.matrix_precision = value;
                    else if (flag == "--tol") automatic.tol = std::stod(value);
                    else if (flag == "--time-limit") automatic.time_limit = std::stod(value);
                    else if (flag == "--iterations") automatic.iteration_limit = std::stoll(value);
                    else if (flag == "--node-limit") automatic.node_limit = std::stoll(value);
                    else if (flag == "--scaling-passes") automatic.scaling_passes = std::stoi(value);
                    else throw std::runtime_error("Unsupported analyze option: " + flag);
                }
                if (automatic.device != "auto" && automatic.device != "cpu" && automatic.device != "cuda" && automatic.device != "hip")
                    throw std::runtime_error("Unknown device");
                const std::vector<std::string> valid_methods={"auto","simplex","dual-simplex","barrier","concurrent","pdhg","halpern","rhpdhg","r2hpdhg"};
                if(std::find(valid_methods.begin(),valid_methods.end(),automatic.method)==valid_methods.end())
                    throw std::runtime_error("Unknown method");
            }
            if (cmd != "inspect") {
                auto hardware = hardware_info();
                auto advisor = advise_model(m, automatic, hardware);
                j["analysis"] = analyze_model(m, automatic);
                j["automatic_selection"] = {{"method", advisor.method}, {"device", advisor.device},
                    {"reason", advisor.reason}, {"available_gpu_bytes", hardware.free_gpu_bytes},
                    {"estimated_gpu_bytes", advisor.estimated_gpu_bytes},
                    {"transformed_rows", advisor.transformed_rows},
                    {"coefficient_range", advisor.coefficient_range},
                    {"newton_fill_estimate", advisor.newton_fill_estimate},
                    {"policy", "deterministic structural heuristic; no performance guarantee"}};

                j["method"] = j["analysis"]["selection"]["algorithm"];
                j["gpu_suitability_heuristic"] =
                    m.A.value.size() >= 100000 ? "Worth measuring CPU/GPU crossover"
                                               : "Low: launch and transfer overhead may dominate";
                j["verification"] =
                    "Original-space feasibility, projected stationarity, complementarity and "
                    "conservative relaxation bounds; MIP tree proof is not independently replayed";
                int64_t integers = 0, transformed_rows = 0;
                for (size_t v = 0; v < m.c.size(); ++v) {
                    integers += m.types[v] != VarType::Continuous;
                    transformed_rows +=
                        std::isfinite(m.lb[v]) && std::isfinite(m.ub[v]) && m.lb[v] != m.ub[v];
                }
                for (size_t i = 0; i < m.rl.size(); ++i) {
                    transformed_rows += std::isfinite(m.rl[i]);
                    transformed_rows += std::isfinite(m.ru[i]) && m.ru[i] != m.rl[i];
                }
                std::vector<int64_t> parent(m.c.size());
                for (size_t v = 0; v < parent.size(); ++v)
                    parent[v] = v;
                auto root = [&](int64_t v) {
                    while (parent[v] != v) {
                        parent[v] = parent[parent[v]];
                        v = parent[v];
                    }
                    return v;
                };
                auto join = [&](int64_t a, int64_t b) {
                    a = root(a);
                    b = root(b);
                    if (a != b)
                        parent[b] = a;
                };
                for (int64_t i = 0; i < m.A.rows; ++i)
                    for (auto k = m.A.ptr[i] + 1; k < m.A.ptr[i + 1]; ++k)
                        join(m.A.index[m.A.ptr[i]], m.A.index[k]);
                for (int64_t i = 0; i < m.Q.rows; ++i)
                    for (auto k = m.Q.ptr[i]; k < m.Q.ptr[i + 1]; ++k)
                        join(i, m.Q.index[k]);
                int64_t components = 0;
                for (size_t v = 0; v < parent.size(); ++v)
                    components += root(v) == int64_t(v);
                j["structure"] = {
                    {"integer_variables", integers},
                    {"quadratic_nonzeros", m.Q.value.size()},
                    {"variable_components", components},
                    {"component_scope",
                     "constraint and quadratic incidence; no automatic decomposition"},
                    {"transformed_row_estimate", transformed_rows}};
                j["recommended_configuration"] = {
                    {"method", advisor.method},
                    {"device", advisor.device},
                    {"branching", automatic.branching},
                    {"cuts", automatic.cuts},
                    {"primal_heuristic", automatic.primal_heuristic},
                    {"scaling", automatic.scaling},
                    {"precision", automatic.matrix_precision}};
                j["advisor_scope"] =
                    "Structural rule-based guidance, not a learned runtime prediction";
            }
            std::cout << j.dump(2) << '\n';
            return 0;
        }
        if (cmd == "sensitivity") {
            Options o;
            o.method = "auto";
            int first = 3;
            std::vector<double> x;
            bool given = argc > 3 && std::string(argv[3]).rfind("--", 0) != 0;
            if (given) {
                x = read_solution(m, argv[3]).x;
                first = 4;
            }
            analysis_options(first, o, nullptr);
            auto report = nlohmann::json::parse(sensitivity_json(m, o, given ? &x : nullptr));
            std::cout << report.dump(2) << '\n';
            return report.at("status") == "OPTIMAL_BASIS" ? 0 : 2;
        }
        if (cmd == "iis") {
            Options o;
            analysis_options(3, o, nullptr);
            auto report = nlohmann::json::parse(iis_json(m, o));
            std::cout << report.dump(2) << '\n';
            return report.at("status") == "IIS_FOUND" ? 0 : 2;
        }
        if (cmd == "differentiate") {
            Options o;
            std::vector<double> upstream;
            int samples = 0;
            double sigma = 0.05;
            uint64_t seed = 1;
            for (int i = 3; i < argc; ++i) {
                std::string flag = argv[i];
                if (i + 1 >= argc) throw std::runtime_error("Missing option value for " + flag);
                std::string value = argv[++i];
                if (flag == "--upstream") {
                    std::ifstream in(value);
                    if (!in) throw std::runtime_error("Cannot read " + value);
                    auto g = nlohmann::json::parse(in);
                    if (g.is_array()) upstream = g.get<std::vector<double>>();
                    else {
                        upstream.assign(m.c.size(), 0.);
                        for (auto it = g.begin(); it != g.end(); ++it) {
                            auto at = std::find(m.names.begin(), m.names.end(), it.key());
                            if (at == m.names.end()) throw std::runtime_error("Unknown variable " + it.key());
                            upstream[at - m.names.begin()] = it.value().get<double>();
                        }
                    }
                } else if (flag == "--samples") samples = std::stoi(value);
                else if (flag == "--sigma") sigma = std::stod(value);
                else if (flag == "--seed") seed = std::stoull(value);
                else if (flag == "--time-limit") o.time_limit = std::stod(value);
                else throw std::runtime_error("Unsupported option for differentiate: " + flag);
            }
            if (samples < 0 || samples > 10000 || !(sigma > 0))
                throw std::runtime_error("Invalid --samples or --sigma");
            auto report = nlohmann::json::parse(differentiate_json(m, o, upstream, samples, sigma, seed));
            std::cout << report.dump(2) << '\n';
            return report.at("status") == "OPTIMAL_BASIS" ? 0 : 2;
        }
        if (cmd == "decompose") {
            Options o;
            std::vector<std::string> linking;
            bool detect_only = false;
            for (int i = 3; i < argc; ++i) {
                std::string flag = argv[i];
                if (flag == "--detect-only") { detect_only = true; continue; }
                if (i + 1 >= argc) throw std::runtime_error("Missing option value for " + flag);
                std::string value = argv[++i];
                if (flag == "--linking") {
                    std::stringstream list(value);
                    for (std::string name; std::getline(list, name, ',');)
                        if (!name.empty()) linking.push_back(name);
                } else if (flag == "--time-limit") o.time_limit = std::stod(value);
                else if (flag == "--iterations") o.iteration_limit = std::stoll(value);
                else if (flag == "--tol") o.tol = std::stod(value);
                else throw std::runtime_error("Unsupported option for decompose: " + flag);
            }
            auto report = nlohmann::json::parse(decompose_json(m, o, linking, detect_only));
            std::cout << report.dump(2) << '\n';
            auto status = report.at("status").get<std::string>();
            return status == "OPTIMAL" || status == "DETECTED" ? 0 : 2;
        }
        if (cmd == "recommend") {
            Options o;
            std::string history;
            analysis_options(3, o, &history);
            if (history.empty()) throw std::runtime_error("recommend requires --history FILE");
            std::cout << nlohmann::json::parse(recommend_json(history, m, o)).dump(2) << '\n';
            return 0;
        }
        if (cmd == "convert") {
            if (argc != 4)
                throw std::runtime_error("convert INPUT OUTPUT");
            write_model(m, argv[3]);
            return 0;
        }
        if (cmd == "verify") {
            if (argc != 4)
                throw std::runtime_error("verify MODEL SOLUTION");
            std::ifstream input(argv[3], std::ios::binary | std::ios::ate);
            if (!input || input.tellg() > 64*1024*1024) throw std::runtime_error("Invalid or oversized verification input");
            input.seekg(0); std::string raw((std::istreambuf_iterator<char>(input)), {});
            auto document = nlohmann::json::parse(raw, [](int depth, nlohmann::json::parse_event_t, const nlohmann::json &) { if (depth > 32) throw std::runtime_error("Verification input nesting limit"); return true; });
            if (document.contains("certificate_schema_version")) {
                auto report = nlohmann::json::parse(verify_certificate_json(m, raw));
                std::cout << report.dump(2) << '\n';
                return report.at("valid").get<bool>() ? 0 : 2;
            }
            auto r = read_solution(m, argv[3]);
            if (!r.infeasibility_ray.empty()) {
                Verifier verifier(m);
                r.certificate_margin = verifier.infeasibility_bound(r.infeasibility_ray);
                bool pass = r.certificate_margin > 0 && std::isfinite(r.certificate_margin);
                r.status = pass ? "VERIFIED_INFEASIBLE" : "VERIFICATION_FAILED";
                std::cout << result_json(m, r) << '\n';
                return pass ? 0 : 2;
            }
            auto declared_objective = r.accuracy.objective;
            auto declared_status = r.status;
            r.accuracy = verify(m, r.x, r.y);
            bool pass =
                r.accuracy.finite && r.accuracy.primal <= 1e-6 && r.accuracy.integrality <= 1e-6;
            bool optimal = pass && !m.is_mip() && r.accuracy.kkt <= 1e-6;
            bool objective_matches =
                !std::isfinite(declared_objective) ||
                (r.accuracy.finite && std::abs(declared_objective - r.accuracy.objective) <=
                                          1e-6 * (1 + std::abs(r.accuracy.objective)));
            if (!objective_matches) {
                pass = false;
                optimal = false;
                r.message = "Reported objective does not match independent calculation";
            }
            if (declared_status == "OPTIMAL" && !m.is_mip() && !optimal) {
                pass = false;
                r.message = "Claimed optimal solution fails independent KKT checks";
            }
            r.status = optimal ? "VERIFIED_OPTIMAL"
                       : pass  ? "VERIFIED_FEASIBLE"
                               : "VERIFICATION_FAILED";
            if (m.is_mip())
                r.message = "Standalone verification checks the incumbent; tree optimality is not "
                            "independently replayed";
            std::cout << result_json(m, r) << '\n';
            return pass ? 0 : 2;
        }
        if (cmd != "solve")
            throw std::runtime_error("Unknown command: " + cmd);
        Options o;
        o.method = "auto";
        std::string output, warm, certificate_output, history, learned_reason;
        bool allow_model_change = false, dry_run = false;
        for (int i = 3; i < argc; i++) {
            std::string a = argv[i];
            auto val = [&]() {
                if (++i >= argc)
                    throw std::runtime_error("Missing option value for " + a);
                return std::string(argv[i]);
            };
            if (a == "--auto")
                o.method = "auto";
            else if (a == "--dry-run")
                dry_run = true;
            else if (a == "--device")
                o.device = val();
            else if (a == "--tol")
                o.tol = std::stod(val());
            else if (a == "--time-limit")
                o.time_limit = std::stod(val());
            else if (a == "--iterations")
                o.iteration_limit = std::stoll(val());
            else if (a == "--node-limit")
                o.node_limit = std::stoll(val());
            else if (a == "--mip-gap")
                o.mip_gap = std::stod(val());
            else if (a == "--integer-tol")
                o.integer_tol = std::stod(val());
            else if (a == "--threads")
                o.threads = std::stoi(val());
            else if (a == "--check-every")
                o.check_every = std::stoi(val());
            else if (a == "--scaling-passes")
                o.scaling_passes = std::stoi(val());
            else if (a == "--scaling")
                o.scaling = val();
            else if (a == "--method")
                o.method = val();
            else if (a == "--node-selection")
                o.node_selection = val();
            else if (a == "--branching")
                o.branching = val();
            else if (a == "--branching-model")
                o.branching_model = val();
            else if (a == "--primal-weight")
                o.primal_weight = val();
            else if (a == "--power-iterations")
                o.power_iterations = std::stoi(val());
            else if (a == "--cuda-graphs")
                o.cuda_graphs = true;
            else if (a == "--batch-strong-branching")
                o.batch_strong_branching = true;
            else if (a == "--gpu-presolve")
                o.gpu_presolve = true;
            else if (a == "--gpu-monitor")
                o.gpu_monitor = true;
            else if (a == "--gpu-indices")
                o.gpu_indices = val();
            else if (a == "--matrix-precision")
                o.matrix_precision = val();
            else if (a == "--newton-precision")
                o.newton_precision = val();
            else if (a == "--polishing")
                o.polishing = true;
            else if (a == "--primal-heuristic")
                o.primal_heuristic = val();
            else if (a == "--cuts")
                o.cuts = true;
            else if (a == "--json-out" || a == "--solution-out")
                output = val();
            else if (a == "--certificate-out" || a == "--certificate")
                certificate_output = val();
            else if (a == "--checkpoint-out")
                o.checkpoint_path = val();
            else if (a == "--resume")
                o.resume_path = val();
            else if (a == "--checkpoint-nodes")
                o.checkpoint_nodes = std::stoll(val());
            else if (a == "--warm-start")
                warm = val();
            else if (a == "--history")
                history = val();
            else if (a == "--allow-model-change")
                allow_model_change = true;
            else if (a == "--verbose")
                o.verbose = true;
            else if (a == "--no-presolve")
                o.presolve = false;
            else if (a == "--no-restart")
                o.restart = false;
            else if (a == "--no-adaptive")
                o.adaptive = false;
            else
                throw std::runtime_error("Unknown option: " + a);
        }
        bool learned = o.method == "learned";
        if (learned) {
            if (history.empty())
                throw std::runtime_error("--method learned requires --history FILE");
            auto method = learned_method(history, m, &learned_reason);
            o.method = method.empty() ? "auto" : method;
        }
        if (dry_run) {
            std::cout << analyze_model(m, o).dump(2) << '\n';
            return 0;
        }
        if (!warm.empty()) {
            auto r = read_solution(m, warm, allow_model_change);
            o.initial_x = r.x;
            o.initial_y = r.y;
            o.initial_basis = r.basis;
            o.basis_fingerprint = r.basis_fingerprint;
        }
        auto r = solve(m, o);
        if (!history.empty())
            record_history(history, m, o, r);
        auto j = nlohmann::json::parse(result_json(m, r));
        j["assurance"] = assurance(m, r, o);
        if (learned)
            j["selection"]["learned"] = {{"method", o.method}, {"reason", learned_reason}};
        j["performance"]["parse_seconds"] = parse_seconds;
        j["selection"]["requested_method"] = o.method;
        j["selection"]["automatic"] = (o.method == "auto" || o.device == "auto");
        j["selection"]["configuration"] = {{"scaling",o.scaling},{"branching",o.branching},
            {"node_selection",o.node_selection},{"scaling_passes",o.scaling_passes},{"cuts_enabled",o.cuts},
            {"primal_heuristic",o.primal_heuristic},{"precision",o.matrix_precision},
            {"tolerance",o.tol},{"time_limit_seconds",o.time_limit},
            {"iteration_limit",o.iteration_limit},{"node_limit",o.node_limit}};
        j["performance"]["end_to_end_seconds"] = r.seconds + parse_seconds;
        j["options"] = {{"tolerance", o.tol},
                        {"time_limit", o.time_limit},
                        {"iteration_limit", o.iteration_limit},
                        {"threads", o.threads},
                        {"scaling_passes", o.scaling_passes},
                        {"scaling", o.scaling},
                        {"method", o.method},
                        {"branching", o.branching},
                        {"node_selection", o.node_selection},
                        {"primal_weight", o.primal_weight},
                        {"power_iterations", o.power_iterations},
                        {"cuda_graphs", o.cuda_graphs},
                        {"gpu_monitor", o.gpu_monitor},
                        {"gpu_indices", o.gpu_indices},
                        {"matrix_precision", o.matrix_precision},
                        {"newton_precision", o.newton_precision},
                        {"polishing", o.polishing},
                        {"cuts", o.cuts},
                        {"primal_heuristic", o.primal_heuristic},
                        {"restart", o.restart},
                        {"adaptive", o.adaptive}};
        j["verification_evidence"] = {
            {"model_fingerprint", m.fingerprint()},
            {"tolerance", o.tol},
            {"scope", m.is_mip() ? "incumbent feasibility; tree bound is solver telemetry"
                                 : "original-space KKT and dual bound"},
            {"independent_final_check", r.accuracy.finite},
            {"tree_optimality_replayed", false}};
        j["verification_certificate"] = nlohmann::json::parse(certificate_json(m, r, o));
        if (!certificate_output.empty()) {
            std::ofstream cert(certificate_output);
            if (!cert)
                throw std::runtime_error("Cannot write certificate");
            cert << j["verification_certificate"].dump(2) << '\n';
        }
        if (!output.empty()) {
            std::ofstream f(output);
            if (!f)
                throw std::runtime_error("Cannot write " + output);
            f << j.dump(2) << '\n';
        }
        std::cout << j.dump(2) << '\n';
        return r.status == "OPTIMAL" ? 0 : 2;
    } catch (const std::exception &ex) {
        std::cerr << nlohmann::json({{"status", "ERROR"}, {"message", ex.what()}}).dump() << '\n';
        return 1;
    }
}
