#include "internal.hpp"
#include <stdexcept>
namespace vantage {
bool cuda_available() {
    return false;
}
std::string cuda_description() {
    return "CUDA backend not compiled";
}
std::unique_ptr<IterationBackend> cuda_backend(const Model &, const std::vector<double> &,
                                               const std::vector<double> &) {
    throw std::runtime_error("CUDA backend not compiled; rebuild with -DVANTAGE_CUDA=ON");
}
} // namespace vantage
