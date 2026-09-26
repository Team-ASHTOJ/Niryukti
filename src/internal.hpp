#pragma once
#include "vantage/vantage.hpp"
#include <chrono>
namespace vantage {
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Prepared {
    Model model;
    std::vector<int64_t> cols, rows;
    std::vector<double> fixed, column_scale, row_scale;
    double objective_scale = 1;
    std::string failure, reason;
    std::vector<double> restore_x(const std::vector<double> &) const;
    std::vector<double> restore_y(const std::vector<double> &, size_t original_rows) const;
};
Prepared prepare(const Model &, const Options &);
double power_norm(const Sparse &, int iterations);
Model dual_feasibility_model(const Model &);
int add_binary_cuts(Model &, int limit);
Model distance_projection_model(const Model &, const std::vector<double> &target);
class PrimalWeightController {
    double integral = 0, previous = 0;
    bool initialized = false;

  public:
    double update(double weight, long double primal_square, long double dual_square);
};
class IterationBackend {
  public:
    virtual ~IterationBackend() = default;
    virtual int advance(int count, double tau, double sigma) = 0;
    virtual int64_t rejected_steps() const {
        return 0;
    }
    virtual bool restart_requested() const {
        return false;
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
                                               const std::vector<double> &, bool adaptive = false);
} // namespace vantage
