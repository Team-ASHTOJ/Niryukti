#pragma once
// Bipartite graph neural network for branching-variable selection (Gasse et al., NeurIPS
// 2019: "Exact combinatorial optimization with graph convolutional neural networks").
// Nodes are the constraints and variables of the node LP; edges are nonzeros weighted by
// |a_ij| / max_k |a_ik|. A constraint update and a variable update are followed by a linear
// scoring head over the fractional candidates. Weights are trained by imitation of
// strong-branching decisions.
#include "vantage/vantage.hpp"
#include <string>
#include <vector>
namespace vantage {
struct BranchingGraph {
    int64_t rows = 0, columns = 0;
    std::vector<int64_t> row_ptr, row_index, col_ptr, col_index; // CSR and CSC edges
    std::vector<double> row_weight, col_weight;
    std::vector<double> variable, constraint; // row-major feature matrices
    std::vector<int64_t> candidates;
};
class BranchingGnn {
  public:
    static constexpr int fv = 10, fc = 4;
    int hidden = 16;
    std::vector<double> theta; // [Wc | bc | Wv | bv | w | b]
    std::string source = "untrained";
    BranchingGnn() = default;
    BranchingGnn(int hidden_units, uint64_t seed);
    // Empty path: built-in trained weights compiled into the engine.
    static BranchingGnn load(const std::string &path);
    std::string to_json(const std::string &metrics_json = "{}") const;
    static BranchingGraph structure(const Model &);
    // Node features from the LP solution. `y` follows the solver convention (c + A'y).
    static void features(BranchingGraph &, const Model &, const std::vector<double> &x,
                         const std::vector<double> &y, double integer_tol);
    std::vector<double> scores(const BranchingGraph &) const;
    // Adds d/dtheta of -log sum_{t in targets} softmax(scores)_t into `gradient` (ties in
    // the expert score are all correct); returns the loss.
    double backward(const BranchingGraph &, const std::vector<size_t> &targets,
                    std::vector<double> &gradient) const;

  private:
    size_t wc() const { return 0; }
    size_t bc() const { return wc() + size_t(hidden) * (fv + fc); }
    size_t wv() const { return bc() + hidden; }
    size_t bv() const { return wv() + size_t(hidden) * (fv + hidden); }
    size_t w() const { return bv() + hidden; }
    size_t b() const { return w() + hidden; }
    void forward(const BranchingGraph &, std::vector<double> &cin, std::vector<double> &zc,
                 std::vector<double> &hc, std::vector<double> &vin, std::vector<double> &zv,
                 std::vector<double> &score) const;
};
std::string train_branching_json(const std::vector<std::string> &paths, const std::string &output,
                                 const Options &, int dives, int epochs, uint64_t seed);
} // namespace vantage
