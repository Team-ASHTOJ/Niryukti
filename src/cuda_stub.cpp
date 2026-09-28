#include "internal.hpp"
#include <stdexcept>
namespace vantage {
bool cuda_propagate_integer_bounds(const Model &, std::vector<double> &, std::vector<double> &,
                                   int) {
    throw std::runtime_error("GPU propagation requires a CUDA build");
}
std::vector<Result> cuda_batch_relaxations(const Model &, const std::vector<std::vector<double>> &,
                                           const std::vector<std::vector<double>> &,
                                           const Options &) {
    throw std::runtime_error("Batched relaxations require a CUDA build");
}
std::vector<double> cuda_linear_solve(const Sparse &, const std::vector<double> &,
                                      const Options &) {
    throw std::runtime_error("GPU Newton solve requires CUDA");
}
Hardware hardware_info() { return {}; }
bool cuda_available() {
    return false;
}
std::string cuda_description() {
    return "CUDA backend not compiled";
}
std::unique_ptr<IterationBackend> cuda_backend(const Model &, const std::vector<double> &,
                                               const std::vector<double> &, bool, bool, double,
                                               bool, bool, const std::string &,
                                               const std::string &) {
    throw std::runtime_error("CUDA backend not compiled; rebuild with -DVANTAGE_CUDA=ON");
}
} // namespace vantage
