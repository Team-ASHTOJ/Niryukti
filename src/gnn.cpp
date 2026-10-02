#include "gnn.hpp"
#include "internal.hpp"
#include "json.hpp"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
namespace vantage {
namespace {
using json = nlohmann::json;
#if __has_include("gnn_weights.inc")
#include "gnn_weights.inc" // defines builtin_branching_weights (trained, see features.md)
#define NIRYUKTI_HAS_BUILTIN_GNN 1
#endif
double relu(double v) { return v > 0 ? v : 0; }
} // namespace

BranchingGnn::BranchingGnn(int hidden_units, uint64_t seed) : hidden(hidden_units) {
    theta.assign(b() + 1, 0.);
    std::mt19937_64 rng(seed);
    auto fill = [&](size_t offset, size_t count, int fan_in) {
        std::normal_distribution<double> normal(0., std::sqrt(2. / fan_in));
        for (size_t k = 0; k < count; ++k)
            theta[offset + k] = normal(rng);
    };
    fill(wc(), size_t(hidden) * (fv + fc), fv + fc);
    fill(wv(), size_t(hidden) * (fv + hidden), fv + hidden);
    fill(w(), hidden, hidden);
    source = "random-initialization";
}

BranchingGnn BranchingGnn::load(const std::string &path) {
    json j;
    if (path.empty()) {
#ifdef NIRYUKTI_HAS_BUILTIN_GNN
        j = json::parse(builtin_branching_weights);
#else
        BranchingGnn g(16, 1);
        g.source = "untrained built-in default";
        return g;
#endif
    } else {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("Cannot read branching model " + path);
        in >> j;
    }
    if (j.value("format", "") != "niryukti-branching-gnn" || j.value("fv", 0) != fv ||
        j.value("fc", 0) != fc)
        throw std::runtime_error("Incompatible branching model");
    BranchingGnn g;
    g.hidden = j.at("hidden").get<int>();
    g.theta = j.at("theta").get<std::vector<double>>();
    if (g.hidden < 1 || g.theta.size() != g.b() + 1)
        throw std::runtime_error("Branching model parameter count mismatch");
    for (double v : g.theta)
        if (!std::isfinite(v))
            throw std::runtime_error("Nonfinite branching model parameter");
    g.source = path.empty() ? "built-in trained weights" : path;
    return g;
}

std::string BranchingGnn::to_json(const std::string &metrics) const {
    return json({{"format", "niryukti-branching-gnn"},
                 {"version", 1},
                 {"hidden", hidden},
                 {"fv", fv},
                 {"fc", fc},
                 {"architecture", "bipartite GNN: constraint update, variable update, linear head"},
                 {"metrics", json::parse(metrics)},
                 {"theta", theta}})
        .dump();
}

BranchingGraph BranchingGnn::structure(const Model &m) {
    BranchingGraph g;
    g.rows = m.A.rows;
    g.columns = m.A.cols;
    g.row_ptr = m.A.ptr;
    g.row_index = m.A.index;
    g.row_weight.resize(m.A.value.size());
    for (int64_t i = 0; i < m.A.rows; ++i) {
        double largest = 0;
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
            largest = std::max(largest, std::abs(m.A.value[k]));
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
            g.row_weight[k] = largest > 0 ? std::abs(m.A.value[k]) / largest : 0;
    }
    // Column view of the normalized weights.
    g.col_ptr.assign(g.columns + 1, 0);
    for (auto j : g.row_index)
        ++g.col_ptr[j + 1];
    for (int64_t j = 0; j < g.columns; ++j)
        g.col_ptr[j + 1] += g.col_ptr[j];
    g.col_index.resize(g.row_index.size());
    g.col_weight.resize(g.row_index.size());
    auto next = g.col_ptr;
    for (int64_t i = 0; i < g.rows; ++i)
        for (auto k = g.row_ptr[i]; k < g.row_ptr[i + 1]; ++k) {
            auto j = g.row_index[k];
            g.col_index[next[j]] = i;
            g.col_weight[next[j]++] = g.row_weight[k];
        }
    return g;
}

void BranchingGnn::features(BranchingGraph &g, const Model &m, const std::vector<double> &x,
                            const std::vector<double> &y, double integer_tol) {
    const int64_t n = g.columns, rows = g.rows;
    double cmax = 0;
    for (double c : m.c)
        cmax = std::max(cmax, std::abs(c));
    if (cmax == 0)
        cmax = 1;
    std::vector<double> rc(m.c.begin(), m.c.end());
    if (y.size() == size_t(rows))
        for (int64_t i = 0; i < rows; ++i)
            for (auto k = g.row_ptr[i]; k < g.row_ptr[i + 1]; ++k)
                rc[g.row_index[k]] += m.A.value[k] * y[i];
    int64_t max_col = 1, max_row = 1;
    for (int64_t j = 0; j < n; ++j)
        max_col = std::max(max_col, g.col_ptr[j + 1] - g.col_ptr[j]);
    for (int64_t i = 0; i < rows; ++i)
        max_row = std::max(max_row, g.row_ptr[i + 1] - g.row_ptr[i]);
    // Cost per unit of column weight (profit density) and share of tight rows per column.
    std::vector<double> ratio(n, 0.), tight_share(n, 0.), activity(rows, 0.);
    for (int64_t i = 0; i < rows; ++i)
        for (auto k = g.row_ptr[i]; k < g.row_ptr[i + 1]; ++k)
            activity[i] += m.A.value[k] * x[g.row_index[k]];
    std::vector<char> tight_row(rows, 0);
    for (int64_t i = 0; i < rows; ++i) {
        double a = activity[i];
        tight_row[i] = (std::isfinite(m.rl[i]) && std::abs(a - m.rl[i]) <= 1e-6 * (1 + std::abs(a))) ||
                       (std::isfinite(m.ru[i]) && std::abs(a - m.ru[i]) <= 1e-6 * (1 + std::abs(a)));
    }
    double ratio_max = 0;
    for (int64_t j = 0; j < n; ++j) {
        double weight = 0;
        int64_t tight = 0, degree = g.col_ptr[j + 1] - g.col_ptr[j];
        for (auto k = g.col_ptr[j]; k < g.col_ptr[j + 1]; ++k) {
            weight += g.col_weight[k];
            tight += tight_row[g.col_index[k]];
        }
        ratio[j] = std::abs(m.c[j]) / std::max(weight, 1e-12);
        ratio_max = std::max(ratio_max, ratio[j]);
        tight_share[j] = degree ? double(tight) / double(degree) : 0.;
    }
    if (ratio_max == 0)
        ratio_max = 1;
    g.variable.assign(size_t(n) * fv, 0.);
    g.candidates.clear();
    for (int64_t j = 0; j < n; ++j) {
        double v = std::clamp(x[j], m.lb[j], m.ub[j]);
        bool integer = m.types[j] != VarType::Continuous;
        double f = v - std::floor(v), distance = std::min(f, 1 - f);
        double *row = &g.variable[size_t(j) * fv];
        row[0] = m.c[j] / cmax;
        row[1] = integer ? f : 0;
        row[2] = integer ? distance : 0;
        row[3] = integer ? 1 : 0;
        row[4] = std::isfinite(m.lb[j]) && std::abs(v - m.lb[j]) <= 1e-9 * (1 + std::abs(v));
        row[5] = std::isfinite(m.ub[j]) && std::abs(v - m.ub[j]) <= 1e-9 * (1 + std::abs(v));
        row[6] = std::clamp(rc[j] / cmax, -5., 5.);
        row[7] = std::log1p(double(g.col_ptr[j + 1] - g.col_ptr[j])) / std::log1p(double(max_col));
        row[8] = ratio[j] / ratio_max;
        row[9] = tight_share[j];
        if (integer && m.lb[j] < m.ub[j] && distance > integer_tol)
            g.candidates.push_back(j);
    }
    double ymax = 0;
    if (y.size() == size_t(rows))
        for (double v : y)
            ymax = std::max(ymax, std::abs(v));
    g.constraint.assign(size_t(rows) * fc, 0.);
    for (int64_t i = 0; i < rows; ++i) {
        long double activity = 0;
        for (auto k = g.row_ptr[i]; k < g.row_ptr[i + 1]; ++k)
            activity += (long double)m.A.value[k] * x[g.row_index[k]];
        double a = double(activity);
        bool tight = (std::isfinite(m.rl[i]) && std::abs(a - m.rl[i]) <= 1e-6 * (1 + std::abs(a))) ||
                     (std::isfinite(m.ru[i]) && std::abs(a - m.ru[i]) <= 1e-6 * (1 + std::abs(a)));
        double *row = &g.constraint[size_t(i) * fc];
        row[0] = ymax > 0 && y.size() == size_t(rows) ? y[i] / ymax : 0;
        row[1] = tight;
        row[2] = m.rl[i] == m.ru[i];
        row[3] = std::log1p(double(g.row_ptr[i + 1] - g.row_ptr[i])) / std::log1p(double(max_row));
    }
}

void BranchingGnn::forward(const BranchingGraph &g, std::vector<double> &cin,
                           std::vector<double> &zc, std::vector<double> &hc,
                           std::vector<double> &vin, std::vector<double> &zv,
                           std::vector<double> &score) const {
    const int ci = fv + fc, vi = fv + hidden, h = hidden;
    cin.assign(size_t(g.rows) * ci, 0.);
    zc.assign(size_t(g.rows) * h, 0.);
    hc.assign(size_t(g.rows) * h, 0.);
    for (int64_t i = 0; i < g.rows; ++i) {
        double *in = &cin[size_t(i) * ci];
        auto degree = g.row_ptr[i + 1] - g.row_ptr[i];
        for (auto k = g.row_ptr[i]; k < g.row_ptr[i + 1]; ++k) {
            const double *xv = &g.variable[size_t(g.row_index[k]) * fv];
            for (int f = 0; f < fv; ++f)
                in[f] += g.row_weight[k] * xv[f];
        }
        if (degree)
            for (int f = 0; f < fv; ++f)
                in[f] /= double(degree);
        for (int f = 0; f < fc; ++f)
            in[fv + f] = g.constraint[size_t(i) * fc + f];
        for (int u = 0; u < h; ++u) {
            double z = theta[bc() + u];
            for (int f = 0; f < ci; ++f)
                z += theta[wc() + size_t(u) * ci + f] * in[f];
            zc[size_t(i) * h + u] = z;
            hc[size_t(i) * h + u] = relu(z);
        }
    }
    const size_t candidates = g.candidates.size();
    vin.assign(candidates * vi, 0.);
    zv.assign(candidates * h, 0.);
    score.assign(candidates, 0.);
    for (size_t t = 0; t < candidates; ++t) {
        auto j = g.candidates[t];
        double *in = &vin[t * vi];
        for (int f = 0; f < fv; ++f)
            in[f] = g.variable[size_t(j) * fv + f];
        auto degree = g.col_ptr[j + 1] - g.col_ptr[j];
        for (auto k = g.col_ptr[j]; k < g.col_ptr[j + 1]; ++k)
            for (int u = 0; u < h; ++u)
                in[fv + u] += g.col_weight[k] * hc[size_t(g.col_index[k]) * h + u];
        if (degree)
            for (int u = 0; u < h; ++u)
                in[fv + u] /= double(degree);
        double s = theta[b()];
        for (int u = 0; u < h; ++u) {
            double z = theta[bv() + u];
            for (int f = 0; f < vi; ++f)
                z += theta[wv() + size_t(u) * vi + f] * in[f];
            zv[t * h + u] = z;
            s += theta[w() + u] * relu(z);
        }
        score[t] = s;
    }
}

std::vector<double> BranchingGnn::scores(const BranchingGraph &g) const {
    std::vector<double> cin, zc, hc, vin, zv, score;
    forward(g, cin, zc, hc, vin, zv, score);
    return score;
}

double BranchingGnn::backward(const BranchingGraph &g, const std::vector<size_t> &targets,
                              std::vector<double> &grad) const {
    std::vector<double> cin, zc, hc, vin, zv, score;
    forward(g, cin, zc, hc, vin, zv, score);
    const int ci = fv + fc, vi = fv + hidden, h = hidden;
    const size_t candidates = score.size();
    double top = *std::max_element(score.begin(), score.end());
    double total = 0;
    for (double s : score)
        total += std::exp(s - top);
    std::vector<char> good(candidates, 0);
    double correct = 0;
    for (auto t : targets) {
        good[t] = 1;
    }
    for (size_t t = 0; t < candidates; ++t)
        if (good[t])
            correct += std::exp(score[t] - top);
    double loss = -(std::log(correct) - std::log(total));
    std::vector<double> dhc(size_t(g.rows) * h, 0.);
    for (size_t t = 0; t < candidates; ++t) {
        double p = std::exp(score[t] - top);
        double ds = p / total - (good[t] ? p / correct : 0.);
        grad[b()] += ds;
        std::vector<double> dz(h, 0.);
        for (int u = 0; u < h; ++u) {
            double z = zv[t * h + u];
            grad[w() + u] += ds * relu(z);
            dz[u] = z > 0 ? ds * theta[w() + u] : 0;
        }
        std::vector<double> dm(h, 0.);
        for (int u = 0; u < h; ++u) {
            if (dz[u] == 0)
                continue;
            grad[bv() + u] += dz[u];
            for (int f = 0; f < vi; ++f)
                grad[wv() + size_t(u) * vi + f] += dz[u] * vin[t * vi + f];
            for (int q = 0; q < h; ++q)
                dm[q] += theta[wv() + size_t(u) * vi + fv + q] * dz[u];
        }
        auto j = g.candidates[t];
        auto degree = g.col_ptr[j + 1] - g.col_ptr[j];
        if (!degree)
            continue;
        for (auto k = g.col_ptr[j]; k < g.col_ptr[j + 1]; ++k)
            for (int q = 0; q < h; ++q)
                dhc[size_t(g.col_index[k]) * h + q] += g.col_weight[k] * dm[q] / double(degree);
    }
    for (int64_t i = 0; i < g.rows; ++i)
        for (int u = 0; u < h; ++u) {
            double d = zc[size_t(i) * h + u] > 0 ? dhc[size_t(i) * h + u] : 0;
            if (d == 0)
                continue;
            grad[bc() + u] += d;
            for (int f = 0; f < ci; ++f)
                grad[wc() + size_t(u) * ci + f] += d * cin[size_t(i) * ci + f];
        }
    return loss;
}

namespace {
struct Sample {
    size_t graph;
    std::vector<double> variable, constraint, strong;
    std::vector<int64_t> candidates;
    std::vector<size_t> targets;
};
} // namespace

std::string train_branching_json(const std::vector<std::string> &paths, const std::string &output,
                                 const Options &options, int dives, int epochs, uint64_t seed) {
    auto start = Clock::now();
    std::mt19937_64 rng(seed);
    std::vector<Model> models;
    std::vector<BranchingGraph> graphs;
    std::vector<Sample> samples;
    int64_t lp_solves = 0;
    Options lp;
    lp.device = "cpu";
    lp.tol = 1e-8;
    lp.time_limit = 30;
    for (auto &path : paths) {
        auto m = read_model(path);
        if (!m.is_mip() || m.is_qp())
            continue;
        Model relax = m;
        std::fill(relax.types.begin(), relax.types.end(), VarType::Continuous);
        lp.method = relax.A.rows <= simplex_row_limit() ? "simplex" : "auto";
        size_t graph_id = graphs.size();
        graphs.push_back(BranchingGnn::structure(m));
        models.push_back(m);
        for (int dive = 0; dive < dives; ++dive) {
            Model node = relax;
            for (int depth = 0; depth < 25; ++depth) {
                if (elapsed(start) > options.time_limit)
                    break;
                auto r = solve(node, lp);
                ++lp_solves;
                if (r.status != "OPTIMAL")
                    break;
                auto &g = graphs[graph_id];
                Model typed = node;
                typed.types = m.types;
                BranchingGnn::features(g, typed, r.x, r.y, options.integer_tol);
                if (g.candidates.empty())
                    break;
                // Most fractional shortlist keeps strong branching affordable.
                std::vector<int64_t> list = g.candidates;
                std::sort(list.begin(), list.end(), [&](int64_t a, int64_t b) {
                    auto d = [&](int64_t j) { double f = r.x[j] - std::floor(r.x[j]); return std::min(f, 1 - f); };
                    return d(a) > d(b);
                });
                if (list.size() > 24)
                    list.resize(24);
                double parent = r.accuracy.objective, large = 1e3 * (1 + std::abs(parent));
                std::vector<double> strong;
                for (auto j : list) {
                    double gains[2];
                    for (int up = 0; up < 2; ++up) {
                        Model child = node;
                        if (up)
                            child.lb[j] = std::ceil(r.x[j]);
                        else
                            child.ub[j] = std::floor(r.x[j]);
                        auto cr = solve(child, lp);
                        ++lp_solves;
                        gains[up] = cr.status == "OPTIMAL" ? std::max(0., cr.accuracy.objective - parent)
                                    : cr.status == "INFEASIBLE" ? large
                                                                : 0.;
                    }
                    strong.push_back(std::max(gains[0], 1e-6) * std::max(gains[1], 1e-6));
                }
                size_t best = std::max_element(strong.begin(), strong.end()) - strong.begin();
                std::vector<size_t> ties;
                for (size_t t = 0; t < strong.size(); ++t)
                    if (strong[t] >= (1 - 1e-6) * strong[best])
                        ties.push_back(t);
                samples.push_back({graph_id, g.variable, g.constraint, strong, list, ties});
                // Expert-guided dive with exploration.
                std::uniform_real_distribution<double> unit;
                auto pick = unit(rng) < .7 ? list[best] : list[rng() % list.size()];
                if (unit(rng) < .5)
                    node.ub[pick] = std::floor(r.x[pick]);
                else
                    node.lb[pick] = std::ceil(r.x[pick]);
            }
        }
    }
    if (samples.size() < 10)
        throw std::runtime_error("Too few labelled branching samples (" +
                                 std::to_string(samples.size()) + "); add MILP training models");
    std::shuffle(samples.begin(), samples.end(), rng);
    size_t validation = std::max<size_t>(1, samples.size() / 5);
    std::vector<Sample> test(samples.begin(), samples.begin() + validation),
        train(samples.begin() + validation, samples.end());
    BranchingGnn model(16, seed);
    auto view = [&](const Sample &s) -> BranchingGraph & {
        auto &g = graphs[s.graph];
        g.variable = s.variable;
        g.constraint = s.constraint;
        g.candidates = s.candidates;
        return g;
    };
    auto evaluate = [&](const std::vector<Sample> &set) {
        double top1 = 0, top3 = 0, fractional = 0, regret = 0;
        for (auto &s : set) {
            auto score = model.scores(view(s));
            std::vector<size_t> order(score.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return score[a] > score[b]; });
            auto correct = [&](size_t t) {
                return std::find(s.targets.begin(), s.targets.end(), t) != s.targets.end();
            };
            top1 += correct(order[0]);
            bool hit = false;
            for (size_t k = 0; k < std::min<size_t>(3, order.size()); ++k)
                hit = hit || correct(order[k]);
            top3 += hit;
            fractional += correct(0); // shortlist is ordered most-fractional first
            regret += s.strong[order[0]] / std::max(1e-300, s.strong[s.targets[0]]);
        }
        double count = double(std::max<size_t>(1, set.size()));
        return std::array<double, 4>{top1 / count, top3 / count, fractional / count, regret / count};
    };
    // Adam (Kingma & Ba 2015) with minibatches.
    std::vector<double> m1(model.theta.size(), 0.), m2(model.theta.size(), 0.), grad;
    const double rate = 0.01, beta1 = .9, beta2 = .999;
    int64_t step = 0;
    json curve = json::array();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), rng);
        double loss = 0;
        for (size_t first = 0; first < train.size(); first += 16) {
            grad.assign(model.theta.size(), 0.);
            size_t last = std::min(train.size(), first + 16);
            for (size_t k = first; k < last; ++k)
                loss += model.backward(view(train[k]), train[k].targets, grad);
            ++step;
            for (size_t p = 0; p < grad.size(); ++p) {
                double g = grad[p] / double(last - first) + 1e-4 * model.theta[p];
                m1[p] = beta1 * m1[p] + (1 - beta1) * g;
                m2[p] = beta2 * m2[p] + (1 - beta2) * g * g;
                double mh = m1[p] / (1 - std::pow(beta1, double(step)));
                double vh = m2[p] / (1 - std::pow(beta2, double(step)));
                model.theta[p] -= rate * mh / (std::sqrt(vh) + 1e-8);
            }
        }
        if (epoch % 10 == 9 || epoch + 1 == epochs) {
            auto v = evaluate(test);
            curve.push_back({{"epoch", epoch + 1},
                             {"train_loss", loss / double(train.size())},
                             {"validation_top1", v[0]}});
        }
    }
    auto tr = evaluate(train), va = evaluate(test);
    json metrics = {{"samples", samples.size()},
                    {"train_samples", train.size()},
                    {"validation_samples", test.size()},
                    {"lp_solves", lp_solves},
                    {"training_models", models.size()},
                    {"train_top1_vs_strong_branching", tr[0]},
                    {"validation_top1_vs_strong_branching", va[0]},
                    {"validation_top3_vs_strong_branching", va[1]},
                    {"validation_top1_most_fractional_rule", va[2]},
                    {"validation_score_ratio_chosen_over_best", va[3]},
                    {"curve", curve},
                    {"seed", seed},
                    {"seconds", elapsed(start)}};
    model.source = output;
    if (!output.empty()) {
        std::ofstream out(output);
        if (!out)
            throw std::runtime_error("Cannot write " + output);
        out << model.to_json(metrics.dump()) << '\n';
    }
    return json({{"analysis", "BRANCHING_GNN_TRAINING"}, {"output", output}, {"metrics", metrics}})
        .dump(2);
}
} // namespace vantage
