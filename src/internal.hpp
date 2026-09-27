#include <stdexcept>
#pragma once
#include "vantage/vantage.hpp"
#include <chrono>
namespace vantage {
inline bool gpu_request(const Options &o) {
    return o.device == "cuda" || o.device == "hip";
}
inline bool stop_requested(const Options &o) {
    return interrupted || (o.cancellation && o.cancellation->load(std::memory_order_relaxed));
}
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Prepared {
    Model model;
    std::vector<int64_t> cols, rows, lower_rows, upper_rows;
    std::vector<double> fixed, column_scale, row_scale;
    double objective_scale = 1;
    std::string failure, reason;
    std::vector<double> failure_ray;
    std::vector<double> restore_x(const std::vector<double> &) const;
    std::vector<double> restore_y(const std::vector<double> &, size_t original_rows) const;
};
Prepared prepare(const Model &, const Options &);
int64_t simplex_row_limit();
Result solve_simplex(const Model &, const Options &);
std::vector<double> cuda_linear_solve(const Sparse &, const std::vector<double> &, const Options &);
Result solve_barrier(const Model &, const Options &);
Result solve_portfolio(const Model &, const Options &);
double power_norm(const Sparse &, int iterations);
Model dual_feasibility_model(const Model &);
int add_binary_cuts(Model &, int limit);
int add_mir_cuts(Model &, int limit, const std::vector<double> *point = nullptr);
bool cuda_propagate_integer_bounds(const Model &, std::vector<double> &, std::vector<double> &,
                                   int passes = 5);
std::vector<Result> cuda_batch_relaxations(const Model &, const std::vector<std::vector<double>> &,
                                           const std::vector<std::vector<double>> &,
                                           const Options &);
Model distance_projection_model(const Model &, const std::vector<double> &target);
class PrimalWeightController {
    double integral = 0, previous = 0;
    bool initialized = false;

  public:
    double update(double weight, long double primal_square, long double dual_square);
    std::vector<double> state() const {
        return {integral, previous, initialized ? 1. : 0.};
    }
    void restore(const std::vector<double> &v) {
        if (v.size() != 3)
            throw std::runtime_error("PID checkpoint dimensions");
        integral = v[0];
        previous = v[1];
        initialized = v[2] != 0;
    }
};
class IterationBackend {
  public:
    virtual ~IterationBackend() = default;
    virtual int advance(int count, double tau, double sigma) = 0;
    virtual std::vector<std::vector<double>> snapshot() = 0;
    virtual void restore(const std::vector<std::vector<double>> &) = 0;
    virtual int64_t rejected_steps() const {
        return 0;
    }
    virtual bool restart_requested() const {
        return false;
    }
    virtual double monitor() {
        return inf;
    }
    virtual void candidates(std::vector<double> &x, std::vector<double> &y, std::vector<double> &ax,
                            std::vector<double> &ay) = 0;
    virtual void reset(const std::vector<double> &x, const std::vector<double> &y) = 0;
};
std::unique_ptr<IterationBackend> cpu_backend(const Model &, const std::vector<double> &,
                                              const std::vector<double> &, bool adaptive = false,
                                              bool halpern = false, double reflection = 0,
                                              bool fixed_point_restarts = false);
std::unique_ptr<IterationBackend> cuda_backend(const Model &, const std::vector<double> &,
                                               const std::vector<double> &, bool adaptive = false,
                                               bool halpern = false, double reflection = 0,
                                               bool residual_restarts = false, bool graphs = false,
                                               const std::string &indices = "auto",
                                               const std::string &precision = "fp64");
} // namespace vantage
