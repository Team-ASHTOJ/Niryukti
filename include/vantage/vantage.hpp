#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>
namespace vantage {
constexpr double inf = std::numeric_limits<double>::infinity();
struct Entry {
    int64_t row, col;
    double value;
};
struct Sparse {
    int64_t rows = 0, cols = 0;
    std::vector<int64_t> ptr, index;
    std::vector<double> value;
    static Sparse build(int64_t rows, int64_t cols, std::vector<Entry> entries);
    Sparse transpose() const;
    std::vector<double> multiply(const std::vector<double> &x) const;
};
enum class VarType { Continuous, Integer, Binary };
struct Model {
    std::string name = "model";
    Sparse A;
    Sparse Q; // Additional symmetric quadratic matrix; q retains the diagonal shorthand.
    std::vector<double> c, q, lb, ub, rl, ru; // q adds diagonal terms to Q; canonical minimization.
    std::vector<VarType> types;
    std::vector<std::string> names, row_names;
    double offset = 0, sense = 1; // output objective = sense * canonical objective
    void validate() const;
    bool is_mip() const;
    bool is_qp() const;
    std::string fingerprint() const;
    double quadratic_norm_bound() const;
};
struct Options {
    std::string device = "auto";
    double tol = 1e-6, time_limit = 60, mip_gap = 1e-4, integer_tol = 1e-6;
    int64_t iteration_limit = 100000, node_limit = 10000;
    int check_every = 100, scaling_passes = 5, threads = 1;
    bool presolve = true, restart = true, adaptive = true, verbose = false;
    std::vector<double> initial_x, initial_y;
    std::vector<int64_t> initial_basis;
    std::string basis_fingerprint;
    std::shared_ptr<std::atomic<bool>> cancellation;
    std::string checkpoint_path, resume_path;
    int64_t checkpoint_nodes = 100;
    // Experimental, opt-in features; existing PDHG/Ruiz defaults are retained.
    std::string method = "pdhg", scaling = "ruiz", branching = "fractional";
    std::string primal_weight = "displacement";
    int power_iterations = 0;
    bool polishing = false;
    bool cuts = false;
    bool cuda_graphs = false, gpu_monitor = false, batch_strong_branching = false,
         gpu_presolve = false;
    std::string gpu_indices = "auto", matrix_precision = "fp64";
    std::string primal_heuristic = "repair", node_selection = "best-bound";
};
struct Hardware {
    bool gpu_available = false;
    double free_gpu_bytes = 0, total_gpu_bytes = 0;
};
struct Advice {
    std::string method, device, reason;
    int64_t nonzeros = 0, transformed_rows = 0, integer_variables = 0;
    double estimated_gpu_bytes = 0, coefficient_range = 1, newton_fill_estimate = 0;
};
Hardware hardware_info();
Advice advise_model(const Model &, const Options &, const Hardware &);
struct Accuracy {
    double objective = inf, primal = inf, dual = inf, gap = inf, kkt = inf, integrality = inf;
    double primal_absolute = inf, dual_absolute = inf, complementarity = inf, lower_bound = -inf;
    bool finite = false;
};
// Reuses sparse transpose and work vectors across independent candidate checks.
// The model must outlive the verifier and remain unchanged.
class Verifier {
    const Model &model;
    Sparse transpose;
    std::vector<double> activity, transpose_product;

  public:
    explicit Verifier(const Model &model);
    Accuracy evaluate(const std::vector<double> &x, const std::vector<double> &y,
                      bool compute_safe_bound = true);
    double infeasibility_bound(const std::vector<double> &y) const;
};
struct Result {
    std::string status = "UNKNOWN", message, backend = "cpu", device_name = "CPU";
    std::vector<double> x, y, infeasibility_ray;
    std::vector<int64_t> basis;
    std::string basis_fingerprint;
    double certificate_margin = 0;
    Accuracy accuracy;
    double seconds = 0, preprocess_seconds = 0, transfer_seconds = 0, iteration_seconds = 0,
           verification_seconds = 0;
    int64_t iterations = 0, restarts = 0, rejected_steps = 0, nodes = 0, nodes_remaining = 0,
            removed_columns = 0, removed_rows = 0, bounds_tightened = 0;
    double best_bound = -inf, mip_gap = inf;
    int64_t strong_branch_probes = 0;
    int64_t weight_updates = 0, polishing_iterations = 0, polishing_attempts = 0;
    double operator_norm_estimate = 0;
    int64_t cuts_added = 0, cut_rounds = 0, local_cuts_added = 0;
    int64_t conflicts_learned = 0, conflicts_pruned = 0, local_branching_calls = 0;
    int64_t bound_conflicts_learned = 0, bound_conflicts_pruned = 0;
    int64_t monitor_checks = 0, host_candidate_checks = 0, skipped_candidate_checks = 0;
    int gpu_index_bits = 0;
    bool graph_execution = false;
    std::string matrix_precision = "fp64";
    std::string method_selected, relaxation_method_selected, device_reason;
    double estimated_gpu_bytes = 0;
    int64_t pump_rounds = 0, rins_calls = 0, heuristic_nodes = 0;
};
Model read_model(const std::string &path);
void write_model(const Model &, const std::string &path);
std::string result_json(const Model &, const Result &);
void write_solution(const Model &, const Result &, const std::string &path);
Result read_solution(const Model &, const std::string &path, bool allow_model_change = false);
std::string certificate_json(const Model &, const Result &, const Options & = {});
std::string verify_certificate_json(const Model &, const std::string &);
std::string certificate_fingerprint(const Model &);
Accuracy verify(const Model &, const std::vector<double> &x, const std::vector<double> &y);
void validate_options(const Options &);
Result solve(const Model &, const Options & = {});
Result solve_continuous(const Model &, const Options &);
Result solve_mip(const Model &, const Options &);
extern volatile std::sig_atomic_t interrupted;
inline const char *gpu_backend_name() {
#ifdef VANTAGE_HAS_HIP
    return "hip";
#else
    return "cuda";
#endif
}
bool cuda_available();
std::string cuda_description();
} // namespace vantage
