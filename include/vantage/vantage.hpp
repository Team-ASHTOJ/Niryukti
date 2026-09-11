#pragma once
#include <algorithm>
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
    std::vector<double> c, q, lb, ub, rl, ru; // q is the diagonal of Q; canonical minimization.
    std::vector<VarType> types;
    std::vector<std::string> names, row_names;
    double offset = 0, sense = 1; // output objective = sense * canonical objective
    void validate() const;
    bool is_mip() const;
    bool is_qp() const;
    std::string fingerprint() const;
};
struct Options {
    std::string device = "auto";
    double tol = 1e-6, time_limit = 60, mip_gap = 1e-4, integer_tol = 1e-6;
    int64_t iteration_limit = 100000, node_limit = 10000;
    int check_every = 100, scaling_passes = 5, threads = 1;
    bool presolve = true, restart = true, adaptive = true, verbose = false;
    std::vector<double> initial_x, initial_y;
};
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
    double certificate_margin = 0;
    Accuracy accuracy;
    double seconds = 0, preprocess_seconds = 0, transfer_seconds = 0, iteration_seconds = 0,
           verification_seconds = 0;
    int64_t iterations = 0, restarts = 0, rejected_steps = 0, nodes = 0, nodes_remaining = 0,
            removed_columns = 0, removed_rows = 0, bounds_tightened = 0;
    double best_bound = -inf, mip_gap = inf;
};
Model read_model(const std::string &path);
void write_model(const Model &, const std::string &path);
std::string result_json(const Model &, const Result &);
void write_solution(const Model &, const Result &, const std::string &path);
Result read_solution(const Model &, const std::string &path, bool allow_model_change = false);
Accuracy verify(const Model &, const std::vector<double> &x, const std::vector<double> &y);
Result solve(const Model &, const Options & = {});
Result solve_continuous(const Model &, const Options &);
Result solve_mip(const Model &, const Options &);
extern volatile std::sig_atomic_t interrupted;
bool cuda_available();
std::string cuda_description();
} // namespace vantage
