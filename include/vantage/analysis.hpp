#pragma once
// Post-optimal analysis, infeasibility diagnosis, bilinear global optimization and
// solve-history learning. Every entry point returns a JSON document so that the CLI,
// C/Python wrappers and dashboard share one report format.
#include "vantage/vantage.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace vantage {
// Basis-exact LP sensitivity: shadow prices, reduced costs, RHS and objective
// coefficient ranging. MILPs are analysed with integers fixed at the incumbent.
// `x` optionally supplies a previously computed solution; otherwise the model is solved.
std::string sensitivity_json(const Model &, const Options & = {},
                             const std::vector<double> *x = nullptr);
// Derivatives of an LP optimum: objective gradients and, given dl/dx (`upstream`), the
// vector-Jacobian products for bounds, matrix entries and (via `samples` perturbed solves of
// scale `sigma`) costs.
std::string differentiate_json(const Model &, const Options &, const std::vector<double> &upstream,
                               int samples = 0, double sigma = 0.05, uint64_t seed = 1);
// Irreducible infeasible subsystem by verified deletion filtering.
std::string iis_json(const Model &, const Options & = {});
// Spatial branch-and-bound with McCormick relaxations for models whose objective and
// constraints contain bilinear products. Reads the bilinear JSON format.
std::string global_solve_json(const std::string &path, const Options & = {});
// Dantzig-Wolfe decomposition of a bordered block-diagonal LP. `linking_rows` forces the
// border (a trailing '*' matches a name prefix); otherwise it is detected.
std::string decompose_json(const Model &, const Options &,
                           const std::vector<std::string> &linking_rows = {},
                           bool detect_only = false);
// Solve history. A record captures structural features, the configuration and outcome.
void record_history(const std::string &path, const Model &, const Options &, const Result &);
// Ranks methods using verified outcomes of structurally similar historical solves.
std::string recommend_json(const std::string &path, const Model &, const Options & = {});
// Returns the learned method, or an empty string when the history is insufficient.
std::string learned_method(const std::string &path, const Model &, std::string *reason = nullptr);
} // namespace vantage
