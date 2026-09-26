#include "json.hpp"
#include "vantage/vantage.hpp"
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
                << "VANTAGE 0.2 — Vector-Accelerated Numerical Toolkit for Advanced Global "
                   "Optimization\nCommands: solve MODEL, inspect MODEL, explain MODEL, "
                   "verify MODEL SOLUTION, convert INPUT OUTPUT, devices\nSolve: --device "
                   "cpu|cuda|auto --tol 1e-6 --time-limit 60 --iterations 100000\n       "
                   "--json-out result.json --warm-start result.json --threads 1 --verbose\n  "
                   "     --no-presolve --scaling-passes 5 --no-restart --no-adaptive\n       "
                   "--node-limit 10000 --mip-gap 1e-4 --check-every 100\n"
                   "       --scaling ruiz|combined --method pdhg|halpern|rhpdhg|r2hpdhg\n"
                   "       --primal-weight displacement|pid --power-iterations 0 --polishing\n"
                   "       --branching fractional|reliability --cuts\n       --primal-heuristic "
                   "repair|pump|rins|all\n"
                   "       halpern: experimental CPU LP, requires --no-adaptive\n";
            return 0;
        }
        std::string cmd = argv[1];
        if (cmd == "devices") {
            std::cout << "CPU: available\nCUDA: " << cuda_description() << '\n';
            return 0;
        }
        if (argc < 3)
            throw std::runtime_error("Model path required");
        auto parse_start = std::chrono::steady_clock::now();
        auto m = read_model(argv[2]);
        double parse_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - parse_start).count();
        if (cmd == "inspect" || cmd == "explain") {
            nlohmann::json j = {{"name", m.name},
                                {"type", m.is_mip()  ? "MILP"
                                         : m.is_qp() ? "QP"
                                                     : "LP"},
                                {"rows", m.A.rows},
                                {"columns", m.A.cols},
                                {"nonzeros", m.A.value.size()},
                                {"fingerprint", m.fingerprint()},
                                {"gpu_storage_estimate_bytes",
                                 48. * m.A.value.size() + 192. * (m.A.rows + m.A.cols + 2)}};
            if (cmd == "explain") {
                j["method"] = m.is_mip()
                                  ? "Best-bound branch-and-bound using VANTAGE PDHG relaxations"
                              : m.is_qp() ? "PDHG with diagonal quadratic proximal step"
                                          : "Restarted PDHG";
                j["gpu_suitability_heuristic"] =
                    m.A.value.size() >= 100000 ? "Worth measuring CPU/GPU crossover"
                                               : "Low: launch and transfer overhead may dominate";
                j["verification"] = "Original-space feasibility, projected stationarity, "
                                    "complementarity and separable Lagrangian bound";
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
            auto r = read_solution(m, argv[3]);
            if (!r.infeasibility_ray.empty()) {
                Verifier verifier(m);
                r.certificate_margin = verifier.infeasibility_bound(r.infeasibility_ray);
                bool pass = r.certificate_margin > 0 && std::isfinite(r.certificate_margin);
                r.status = pass ? "VERIFIED_INFEASIBLE" : "VERIFICATION_FAILED";
                std::cout << result_json(m, r) << '\n';
                return pass ? 0 : 2;
            }
            r.accuracy = verify(m, r.x, r.y);
            bool pass =
                r.accuracy.finite && r.accuracy.primal <= 1e-6 && r.accuracy.integrality <= 1e-6;
            bool optimal = pass && !m.is_mip() && r.accuracy.kkt <= 1e-6;
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
        std::string output, warm;
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
            else if (a == "--branching")
                o.branching = val();
            else if (a == "--primal-weight")
                o.primal_weight = val();
            else if (a == "--power-iterations")
                o.power_iterations = std::stoi(val());
            else if (a == "--polishing")
                o.polishing = true;
            else if (a == "--primal-heuristic")
                o.primal_heuristic = val();
            else if (a == "--cuts")
                o.cuts = true;
            else if (a == "--json-out" || a == "--solution-out")
                output = val();
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
                        {"primal_weight", o.primal_weight},
                        {"power_iterations", o.power_iterations},
                        {"polishing", o.polishing},
                        {"cuts", o.cuts},
                        {"primal_heuristic", o.primal_heuristic},
                        {"restart", o.restart},
                        {"adaptive", o.adaptive}};
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
