#include "json.hpp"
#include "vantage/vantage.hpp"
#include "session.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace vantage;
int main(int argc, char **argv) {
    std::signal(SIGINT, [](int) { interrupted = 1; });
    try {
        if (argc < 2) {
            std::cout
                << "NIRYUKTI 0.2 — Independent Sparse Optimization Engine\nCommands: solve MODEL, "
                   "inspect MODEL, explain MODEL, session MODEL, "
                   "verify MODEL SOLUTION, convert INPUT OUTPUT, devices\nSolve: --device "
                   "cpu|cuda|auto --tol 1e-6 --time-limit 60 --iterations 100000\n       "
                   "--json-out result.json --warm-start result.json --threads 1 --verbose\n  "
                   "     --no-presolve --scaling-passes 5 --no-restart --no-adaptive\n       "
                   "--node-limit 10000 --mip-gap 1e-4 --check-every 100\n"
                   "       --scaling ruiz|combined --method "
                   "auto|simplex|dual-simplex|barrier|concurrent|pdhg|halpern|rhpdhg|r2hpdhg\n"
                   "       --primal-weight displacement|pid --power-iterations 0 --polishing\n"
                   "       --cuda-graphs --gpu-indices auto|32|64 --matrix-precision fp64|mixed\n"
                   "       --gpu-monitor --certificate-out certificate.json\n"
                   "       --node-selection best-bound|depth-first|best-estimate\n"
                   "       --branching fractional|reliability --cuts\n       --primal-heuristic "
                   "repair|pump|rins|local|all\n"
                   "       --checkpoint-out state.json --checkpoint-nodes 100 --resume state.json\n"
                   "       --gpu-presolve --batch-strong-branching\n"
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
        auto parse_start = std::chrono::steady_clock::now();
        auto m = read_model(argv[2]);
        double parse_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - parse_start).count();
        if (cmd == "inspect" || cmd == "explain") {
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
            if (cmd == "explain") {
                j["method"] =
                    m.is_mip() ? "Best-bound branch-and-bound using NIRYUKTI PDHG relaxations"
                    : m.is_qp()
                        ? (!m.Q.value.empty() ? "Smooth primal-dual splitting with sparse Q*x"
                                              : "PDHG with diagonal quadratic proximal step")
                        : "Restarted PDHG";
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
                bool compact = !m.is_qp() && transformed_rows <= 512;
                bool large = m.A.value.size() + m.Q.value.size() >= 100000;
                j["structure"] = {
                    {"integer_variables", integers},
                    {"quadratic_nonzeros", m.Q.value.size()},
                    {"variable_components", components},
                    {"component_scope",
                     "constraint and quadratic incidence; no automatic decomposition"},
                    {"transformed_row_estimate", transformed_rows}};
                j["recommended_configuration"] = {
                    {"method", compact ? "auto" : "pdhg"},
                    {"device", large && !compact ? "auto" : "cpu"},
                    {"branching",
                     m.is_mip() ? "reliability (opt-in; benchmark first)" : "not applicable"}};
                j["advisor_scope"] =
                    "Structural rule-based guidance, not a learned runtime prediction";
            }
            std::cout << j.dump(2) << '\n';
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
        std::string output, warm, certificate_output;
        bool allow_model_change = false;
        for (int i = 3; i < argc; i++) {
            std::string a = argv[i];
            auto val = [&]() {
                if (++i >= argc)
                    throw std::runtime_error("Missing option value for " + a);
                return std::string(argv[i]);
            };
            if (a == "--device")
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
        if (!warm.empty()) {
            auto r = read_solution(m, warm, allow_model_change);
            o.initial_x = r.x;
            o.initial_y = r.y;
            o.initial_basis = r.basis;
            o.basis_fingerprint = r.basis_fingerprint;
        }
        auto r = solve(m, o);
        auto j = nlohmann::json::parse(result_json(m, r));
        j["performance"]["parse_seconds"] = parse_seconds;
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
