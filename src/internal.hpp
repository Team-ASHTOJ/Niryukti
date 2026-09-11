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
class IterationBackend {
  public:
    virtual ~IterationBackend() = default;
    virtual int advance(int count, double tau, double sigma) = 0;
    virtual int64_t rejected_steps() const {
        return 0;
    }
    virtual void candidates(std::vector<double> &x, std::vector<double> &y, std::vector<double> &ax,
                            std::vector<double> &ay) = 0;
    virtual void reset(const std::vector<double> &x, const std::vector<double> &y) = 0;
};
std::unique_ptr<IterationBackend> cpu_backend(const Model &, const std::vector<double> &,
                                              const std::vector<double> &, bool adaptive = false);
std::unique_ptr<IterationBackend> cuda_backend(const Model &, const std::vector<double> &,
                                               const std::vector<double> &, bool adaptive = false);
} // namespace vantage
