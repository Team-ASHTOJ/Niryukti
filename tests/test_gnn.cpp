// Finite-difference check of the branching GNN's hand-written backpropagation, plus a
// built-in weight load and a deterministic scoring check.
#include "gnn.hpp"
#include <cmath>
#include <cstdio>
#include <random>
using namespace vantage;
// Explicit checks: assert() is compiled out in Release test builds.
#define CHECK(condition)                                                                   \
    do {                                                                                   \
        if (!(condition)) {                                                                \
            std::fprintf(stderr, "check failed: %s (line %d)\n", #condition, __LINE__);    \
            return 1;                                                                      \
        }                                                                                  \
    } while (0)
int main() {
    Model m;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(0, 1);
    const int n = 12, rows = 6;
    std::vector<Entry> entries;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < n; ++j)
            if (u(rng) < .45)
                entries.push_back({i, j, 1 + 9 * u(rng)});
    m.A = Sparse::build(rows, n, entries);
    for (int j = 0; j < n; ++j) {
        m.names.push_back("x" + std::to_string(j));
        m.c.push_back(-(1 + 20 * u(rng)));
        m.q.push_back(0);
        m.lb.push_back(0);
        m.ub.push_back(1);
        m.types.push_back(VarType::Binary);
    }
    for (int i = 0; i < rows; ++i) {
        m.row_names.push_back("r" + std::to_string(i));
        m.rl.push_back(-inf);
        m.ru.push_back(10);
    }
    std::vector<double> x(n), y(rows);
    for (auto &v : x)
        v = .1 + .8 * u(rng);
    for (auto &v : y)
        v = u(rng);
    auto graph = BranchingGnn::structure(m);
    BranchingGnn::features(graph, m, x, y, 1e-6);
    CHECK(graph.candidates.size() == size_t(n));
    BranchingGnn net(8, 5);
    std::vector<size_t> targets = {3, 7};
    std::vector<double> grad(net.theta.size(), 0.), scratch(net.theta.size());
    net.backward(graph, targets, grad);
    double worst = 0;
    int checked = 0;
    for (size_t p = 0; p < net.theta.size(); ++p) {
        double keep = net.theta[p], h = 1e-6;
        net.theta[p] = keep + h;
        double plus = net.backward(graph, targets, scratch);
        net.theta[p] = keep - h;
        double minus = net.backward(graph, targets, scratch);
        net.theta[p] = keep;
        double fd = (plus - minus) / (2 * h);
        double scale = std::abs(fd) + std::abs(grad[p]);
        if (scale > 1e-7) {
            worst = std::max(worst, std::abs(fd - grad[p]) / scale);
            ++checked;
        }
    }
    std::printf("gnn gradient check: %d parameters, worst relative error %.2e\n", checked, worst);
    CHECK(checked > 50 && worst < 1e-5);
    auto builtin = BranchingGnn::load("");
    auto scores = builtin.scores(graph);
    CHECK(scores.size() == graph.candidates.size());
    for (double s : scores)
        CHECK(std::isfinite(s));
    std::printf("built-in branching model: %s\n", builtin.source.c_str());
    return 0;
}
