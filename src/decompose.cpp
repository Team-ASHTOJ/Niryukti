#include "internal.hpp"
#include "json.hpp"
#include "vantage/analysis.hpp"
#include <future>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
// Dantzig-Wolfe decomposition for bordered block-diagonal LPs.
//
// Structure detection removes the densest rows one at a time and keeps the configuration
// whose remaining rows split the variables into the most balanced set of >= 2 blocks with a
// small linking border (a greedy stand-in for hypergraph partitioning; Ferris & Horn 1998,
// Bergner et al. 2015).
//
// The master LP keeps linking rows, one convexity row per block and columns that are block
// extreme points (Dantzig & Wolfe 1960). Phase 1 drives elastic artificials on the linking
// rows to zero; phase 2 optimizes the true objective. Block pricing problems are independent
// and run concurrently. The lower bound is the Lagrangian bound evaluated by the
// independent verifier on the ORIGINAL model with duals assembled from the master (linking
// rows) and the pricing problems (block rows).
namespace vantage {
namespace {
using json = nlohmann::json;
struct Structure {
    std::vector<int64_t> row_block, column_block; // -1 = linking / master-only
    int64_t blocks = 0, linking_rows = 0;
    double score = -inf;
    std::string detector = "user-specified linking rows";
};
Structure components(const Model &m, const std::vector<char> &linking) {
    const int64_t n = int64_t(m.c.size());
    std::vector<int64_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int64_t v) {
        while (parent[v] != v)
            v = parent[v] = parent[parent[v]];
        return v;
    };
    std::vector<char> used(n, 0);
    for (int64_t i = 0; i < m.A.rows; ++i) {
        if (linking[i])
            continue;
        int64_t first = -1;
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k) {
            auto j = m.A.index[k];
            used[j] = 1;
            if (first < 0)
                first = j;
            else {
                auto a = root(first), b = root(j);
                if (a != b)
                    parent[b] = a;
            }
        }
    }
    Structure s;
    s.column_block.assign(n, -1);
    std::map<int64_t, int64_t> id;
    for (int64_t j = 0; j < n; ++j)
        if (used[j]) {
            auto r = root(j);
            if (!id.count(r))
                id[r] = int64_t(id.size());
            s.column_block[j] = id[r];
        }
    s.blocks = int64_t(id.size());
    s.row_block.assign(m.A.rows, -1);
    for (int64_t i = 0; i < m.A.rows; ++i) {
        if (linking[i]) {
            ++s.linking_rows;
            continue;
        }
        if (m.A.ptr[i] < m.A.ptr[i + 1])
            s.row_block[i] = s.column_block[m.A.index[m.A.ptr[i]]];
    }
    return s;
}
double score(const Model &m, Structure &s) {
    if (s.blocks < 2)
        return s.score = -inf;
    // Reward many balanced blocks; penalize border size.
    std::vector<int64_t> size(s.blocks, 0);
    for (auto b : s.column_block)
        if (b >= 0)
            ++size[b];
    double largest = double(*std::max_element(size.begin(), size.end()));
    double total = std::accumulate(size.begin(), size.end(), 0.);
    double balance = total / (largest * double(s.blocks));
    return s.score = std::log2(double(s.blocks)) * balance -
                     4. * double(s.linking_rows) / double(std::max<int64_t>(1, m.A.rows));
}
// Row family = name with digits and separators after the first digit removed
// ("jetty_3" -> "jetty"). Families model the indexed constraint groups of algebraic models.
std::string family(const std::string &name) {
    auto cut = name.find_first_of("0123456789");
    std::string f = name.substr(0, cut);
    while (!f.empty() && (f.back() == '_' || f.back() == '.' || f.back() == '-' || f.back() == '['))
        f.pop_back();
    return f.empty() ? name : f;
}
Structure detect_families(const Model &m) {
    std::map<std::string, std::vector<int64_t>> groups;
    for (int64_t i = 0; i < m.A.rows; ++i)
        groups[family(m.row_names[i])].push_back(i);
    std::vector<char> linking(m.A.rows, 0);
    auto best = components(m, linking);
    score(m, best);
    if (groups.size() < 2 || groups.size() > 200)
        return best;
    std::set<std::string> chosen;
    // Borders often need two families together (e.g. shared supply and shared demand);
    // seed with the best pair when no single family splits the model.
    if (best.blocks < 2 && groups.size() <= 60) {
        std::vector<std::string> names;
        for (auto &[name, rows] : groups)
            names.push_back(name);
        std::pair<std::string, std::string> seed;
        for (size_t a = 0; a < names.size(); ++a)
            for (size_t b = a + 1; b < names.size(); ++b) {
                auto trial = linking;
                for (auto i : groups[names[a]])
                    trial[i] = 1;
                for (auto i : groups[names[b]])
                    trial[i] = 1;
                auto s = components(m, trial);
                score(m, s);
                if (s.score > best.score + 1e-12) {
                    best = s;
                    seed = {names[a], names[b]};
                }
            }
        if (!seed.first.empty())
            for (auto &name : {seed.first, seed.second}) {
                chosen.insert(name);
                for (auto i : groups[name])
                    linking[i] = 1;
            }
    }
    while (true) {
        Structure improved = best;
        std::string pick;
        for (auto &[name, rows] : groups) {
            if (chosen.count(name))
                continue;
            auto trial = linking;
            for (auto i : rows)
                trial[i] = 1;
            auto s = components(m, trial);
            score(m, s);
            if (s.score > improved.score + 1e-12) {
                improved = s;
                pick = name;
            }
        }
        if (pick.empty())
            return best;
        chosen.insert(pick);
        for (auto i : groups[pick])
            linking[i] = 1;
        best = improved;
    }
}
Structure detect(const Model &m, int64_t max_border) {
    std::vector<int64_t> order(m.A.rows);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return m.A.ptr[a + 1] - m.A.ptr[a] > m.A.ptr[b + 1] - m.A.ptr[b];
    });
    std::vector<char> linking(m.A.rows, 0);
    Structure best = components(m, linking);
    score(m, best);
    int64_t limit = std::min<int64_t>(max_border, m.A.rows / 2);
    for (int64_t k = 0; k < limit; ++k) {
        linking[order[k]] = 1;
        auto s = components(m, linking);
        if (score(m, s) > best.score + 1e-12)
            best = s;
    }
    auto named = detect_families(m);
    if (named.score > best.score) {
        named.detector = "row-family border search";
        return named;
    }
    best.detector = "dense-row border search";
    return best;
}
std::vector<char> rows_by_name(const Model &m, const std::vector<std::string> &names) {
    std::vector<char> linking(m.A.rows, 0);
    for (auto &name : names) {
        bool found = false;
        for (int64_t i = 0; i < m.A.rows; ++i)
            if (m.row_names[i] == name || (name.back() == '*' &&
                                           m.row_names[i].rfind(name.substr(0, name.size() - 1), 0) == 0)) {
                linking[i] = 1;
                found = true;
            }
        if (!found)
            throw std::runtime_error("Unknown linking row: " + name);
    }
    return linking;
}
struct Block {
    std::vector<int64_t> columns, rows; // global indices
    Model sub;                          // rows/cols restricted; objective rewritten per pricing
    std::vector<Entry> link;            // (linking position, local column, value)
    std::vector<std::vector<double>> points;
};
Options lp_options(const Options &o) {
    Options lp;
    lp.device = "cpu";
    lp.tol = std::min(o.tol, 1e-8);
    lp.time_limit = o.time_limit;
    lp.method = "auto";
    return lp;
}
} // namespace

std::string decompose_json(const Model &m, const Options &o, const std::vector<std::string> &forced,
                           bool detect_only) {
    auto start = Clock::now();
    if (m.is_qp() || m.is_mip())
        return json({{"analysis", "DANTZIG_WOLFE"}, {"status", "UNSUPPORTED"},
                     {"message", "Decomposition supports continuous LPs"}})
            .dump(2);
    Structure s = forced.empty() ? detect(m, std::max<int64_t>(1, m.A.rows / 4))
                                 : components(m, rows_by_name(m, forced));
    json structure = {{"blocks", s.blocks},
                      {"linking_rows", s.linking_rows},
                      {"detection", s.detector}};
    json linking_names = json::array();
    for (int64_t i = 0; i < m.A.rows; ++i)
        if (s.row_block[i] < 0 && m.A.ptr[i] < m.A.ptr[i + 1])
            linking_names.push_back(m.row_names[i]);
    structure["linking_row_names"] = linking_names;
    std::vector<int64_t> block_sizes(std::max<int64_t>(s.blocks, 0), 0);
    for (auto b : s.column_block)
        if (b >= 0)
            ++block_sizes[b];
    structure["block_columns"] = block_sizes;
    json report = {{"analysis", "DANTZIG_WOLFE"}, {"model", m.name}, {"structure", structure}};
    if (s.blocks < 2) {
        report["status"] = "NO_STRUCTURE";
        report["message"] = "No bordered block-diagonal structure with >= 2 blocks was found";
        return report.dump(2);
    }
    if (detect_only) {
        report["status"] = "DETECTED";
        return report.dump(2);
    }
    const int64_t n = int64_t(m.c.size());
    // Linking rows (empty rows ignored) and master-only columns.
    std::vector<int64_t> link_rows, link_position(m.A.rows, -1), master_columns;
    for (int64_t i = 0; i < m.A.rows; ++i)
        if (s.row_block[i] < 0) {
            link_position[i] = int64_t(link_rows.size());
            link_rows.push_back(i);
        }
    std::vector<int64_t> master_position(n, -1);
    for (int64_t j = 0; j < n; ++j)
        if (s.column_block[j] < 0) {
            master_position[j] = int64_t(master_columns.size());
            master_columns.push_back(j);
        }
    // Unbounded block columns are boxed so every pricing problem has an optimal vertex.
    double scale = 1;
    for (auto *v : {&m.lb, &m.ub, &m.rl, &m.ru})
        for (double b : *v)
            if (std::isfinite(b))
                scale = std::max(scale, std::abs(b));
    const double box = 1e6 * scale;
    bool boxed = false;
    std::vector<Block> blocks(s.blocks);
    std::vector<int64_t> local(n, -1);
    for (int64_t j = 0; j < n; ++j)
        if (s.column_block[j] >= 0) {
            auto &b = blocks[s.column_block[j]];
            local[j] = int64_t(b.columns.size());
            b.columns.push_back(j);
        }
    for (int64_t i = 0; i < m.A.rows; ++i)
        if (s.row_block[i] >= 0)
            blocks[s.row_block[i]].rows.push_back(i);
    for (auto &b : blocks) {
        Model &sub = b.sub;
        sub.name = m.name + "_block";
        for (auto j : b.columns) {
            sub.names.push_back(m.names[j]);
            sub.lb.push_back(std::isfinite(m.lb[j]) ? m.lb[j] : -box);
            sub.ub.push_back(std::isfinite(m.ub[j]) ? m.ub[j] : box);
            boxed = boxed || !std::isfinite(m.lb[j]) || !std::isfinite(m.ub[j]);
        }
        size_t nb = b.columns.size();
        sub.c.assign(nb, 0.);
        sub.q.assign(nb, 0.);
        sub.types.assign(nb, VarType::Continuous);
        std::vector<Entry> e;
        for (size_t r = 0; r < b.rows.size(); ++r) {
            auto i = b.rows[r];
            for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
                e.push_back({int64_t(r), local[m.A.index[k]], m.A.value[k]});
            sub.rl.push_back(m.rl[i]);
            sub.ru.push_back(m.ru[i]);
            sub.row_names.push_back(m.row_names[i]);
        }
        sub.A = Sparse::build(int64_t(b.rows.size()), int64_t(nb), std::move(e));
    }
    for (size_t p = 0; p < link_rows.size(); ++p) {
        auto i = link_rows[p];
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k) {
            auto j = m.A.index[k];
            if (s.column_block[j] >= 0)
                blocks[s.column_block[j]].link.push_back({int64_t(p), local[j], m.A.value[k]});
        }
    }
    const int64_t L = int64_t(link_rows.size()), B = s.blocks;
    auto opts = lp_options(o);
    // Pricing: min (c_b + A_link_b' y_link) x  over the block polyhedron.
    struct Priced {
        std::string status;
        std::vector<double> x, y;
        double value = inf;
    };
    auto price = [&](int64_t b, const std::vector<double> &y_link, bool phase1) {
        auto sub = blocks[b].sub;
        for (size_t k = 0; k < blocks[b].columns.size(); ++k)
            sub.c[k] = phase1 ? 0. : m.c[blocks[b].columns[k]];
        for (auto &e : blocks[b].link)
            sub.c[e.col] += e.value * y_link[e.row];
        auto r = solve(sub, opts);
        Priced p{r.status, r.x, r.y, inf};
        if (r.status == "OPTIMAL") {
            long double v = 0;
            for (size_t k = 0; k < r.x.size(); ++k)
                v += (long double)sub.c[k] * r.x[k];
            p.value = double(v);
        }
        return p;
    };
    auto price_all = [&](const std::vector<double> &y_link, bool phase1) {
        std::vector<std::future<Priced>> jobs;
        for (int64_t b = 0; b < B; ++b)
            jobs.push_back(std::async(std::launch::async, price, b, y_link, phase1));
        std::vector<Priced> out;
        for (auto &j : jobs)
            out.push_back(j.get());
        return out;
    };
    // Initial columns.
    {
        auto first = price_all(std::vector<double>(L, 0.), false);
        for (int64_t b = 0; b < B; ++b) {
            if (first[b].status == "INFEASIBLE") {
                report["status"] = "INFEASIBLE";
                report["message"] = "Block " + std::to_string(b) + " is infeasible on its own";
                return report.dump(2);
            }
            if (first[b].status != "OPTIMAL") {
                report["status"] = "NUMERICAL_ERROR";
                report["message"] = "Initial pricing failed: " + first[b].status;
                return report.dump(2);
            }
            blocks[b].points.push_back(first[b].x);
        }
    }
    auto link_activity = [&](int64_t b, const std::vector<double> &x) {
        std::vector<double> a(L, 0.);
        for (auto &e : blocks[b].link)
            a[e.row] += e.value * x[e.col];
        return a;
    };
    auto block_cost = [&](int64_t b, const std::vector<double> &x) {
        long double v = 0;
        for (size_t k = 0; k < x.size(); ++k)
            v += (long double)m.c[blocks[b].columns[k]] * x[k];
        return double(v);
    };
    int64_t iterations = 0, columns = B, phase = 1;
    double master_objective = inf, best_bound = -inf;
    std::vector<double> lambda, master_x, y_full(m.A.rows, 0.);
    std::string status = "ITERATION_LIMIT";
    json history = json::array();
    const int64_t max_iterations = std::max<int64_t>(50, std::min<int64_t>(o.iteration_limit, 2000));
    while (iterations < max_iterations) {
        if (elapsed(start) > o.time_limit) {
            status = "TIME_LIMIT";
            break;
        }
        ++iterations;
        // Master columns: master-only originals, lambdas, then 2L artificials.
        Model master;
        master.name = m.name + "_master";
        std::vector<Entry> e;
        auto add_column = [&](const std::string &name, double cost, double lb, double ub) {
            master.names.push_back(name);
            master.c.push_back(cost);
            master.q.push_back(0);
            master.lb.push_back(lb);
            master.ub.push_back(ub);
            master.types.push_back(VarType::Continuous);
            return int64_t(master.c.size() - 1);
        };
        for (auto j : master_columns)
            add_column(m.names[j], phase == 1 ? 0. : m.c[j], m.lb[j], m.ub[j]);
        // Master-only column coefficients in linking rows.
        for (size_t p = 0; p < link_rows.size(); ++p) {
            auto i = link_rows[p];
            for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k) {
                auto j = m.A.index[k];
                if (s.column_block[j] < 0)
                    e.push_back({int64_t(p), master_position[j], m.A.value[k]});
            }
        }
        std::vector<std::pair<int64_t, int64_t>> lambda_owner;
        for (int64_t b = 0; b < B; ++b)
            for (size_t k = 0; k < blocks[b].points.size(); ++k) {
                const auto &x = blocks[b].points[k];
                auto col = add_column("lambda_" + std::to_string(b) + "_" + std::to_string(k),
                                      phase == 1 ? 0. : block_cost(b, x), 0, inf);
                auto a = link_activity(b, x);
                for (int64_t p = 0; p < L; ++p)
                    if (a[p] != 0)
                        e.push_back({p, col, a[p]});
                e.push_back({L + b, col, 1});
                lambda_owner.push_back({b, int64_t(k)});
            }
        for (int64_t p = 0; p < L; ++p) {
            auto up = add_column("artificial_up_" + std::to_string(p), phase == 1 ? 1. : 0., 0,
                                 phase == 1 ? inf : 0.);
            auto down = add_column("artificial_down_" + std::to_string(p), phase == 1 ? 1. : 0., 0,
                                   phase == 1 ? inf : 0.);
            e.push_back({p, up, 1});
            e.push_back({p, down, -1});
        }
        for (int64_t p = 0; p < L; ++p) {
            master.rl.push_back(m.rl[link_rows[p]]);
            master.ru.push_back(m.ru[link_rows[p]]);
            master.row_names.push_back(m.row_names[link_rows[p]]);
        }
        for (int64_t b = 0; b < B; ++b) {
            master.rl.push_back(1);
            master.ru.push_back(1);
            master.row_names.push_back("convexity_" + std::to_string(b));
        }
        master.A = Sparse::build(L + B, int64_t(master.c.size()), std::move(e));
        auto r = solve(master, opts);
        if (r.status != "OPTIMAL") {
            status = phase == 1 && r.status == "INFEASIBLE" ? "INFEASIBLE" : "NUMERICAL_ERROR";
            report["message"] = "Restricted master: " + r.status + " " + r.message;
            break;
        }
        master_objective = r.accuracy.objective;
        std::vector<double> y_link(r.y.begin(), r.y.begin() + L);
        auto priced = price_all(y_link, phase == 1);
        bool improved = false, pricing_ok = true;
        for (int64_t b = 0; b < B; ++b) {
            if (priced[b].status != "OPTIMAL") {
                pricing_ok = false;
                continue;
            }
            // Reduced cost of the new column = pricing value + convexity dual.
            double reduced = priced[b].value + r.y[L + b];
            if (reduced < -1e-9 * (1 + std::abs(master_objective))) {
                blocks[b].points.push_back(priced[b].x);
                ++columns;
                improved = true;
            }
        }
        if (phase == 2 && pricing_ok) {
            // Assemble full dual vector: linking rows from master, block rows from pricing.
            std::fill(y_full.begin(), y_full.end(), 0.);
            for (int64_t p = 0; p < L; ++p)
                y_full[link_rows[p]] = y_link[p];
            for (int64_t b = 0; b < B; ++b)
                for (size_t t = 0; t < blocks[b].rows.size(); ++t)
                    y_full[blocks[b].rows[t]] = priced[b].y[t];
            // Primal recovery: x = sum lambda * extreme point + master-only values.
            master_x.assign(n, 0.);
            for (size_t t = 0; t < master_columns.size(); ++t)
                master_x[master_columns[t]] = r.x[t];
            size_t offset = master_columns.size();
            for (size_t t = 0; t < lambda_owner.size(); ++t) {
                auto [b, k] = lambda_owner[t];
                double weight = r.x[offset + t];
                if (weight != 0)
                    for (size_t q = 0; q < blocks[b].columns.size(); ++q)
                        master_x[blocks[b].columns[q]] += weight * blocks[b].points[k][q];
            }
            auto acc = verify(m, master_x, y_full);
            if (std::isfinite(acc.lower_bound) && !boxed)
                best_bound = std::max(best_bound, acc.lower_bound);
            history.push_back({{"iteration", iterations},
                               {"master_objective", m.sense * master_objective},
                               {"verified_bound", std::isfinite(acc.lower_bound)
                                                      ? json(m.sense * acc.lower_bound)
                                                      : json()},
                               {"columns", columns}});
        }
        if (phase == 1) {
            if (master_objective <= 1e-9) {
                phase = 2;
                continue;
            }
            if (!improved && pricing_ok) {
                status = "INFEASIBLE";
                report["message"] = "Phase 1 converged with positive linking violation " +
                                    std::to_string(master_objective);
                break;
            }
            continue;
        }
        if (!improved && pricing_ok) {
            status = "OPTIMAL";
            break;
        }
    }
    report["iterations"] = iterations;
    report["columns_generated"] = columns;
    report["bound_history"] = history;
    report["parallel_pricing_threads"] = B;
    report["seconds"] = elapsed(start);
    if (!master_x.empty()) {
        Verifier verifier(m);
        auto acc = verifier.evaluate(master_x, y_full);
        double objective = acc.objective;
        report["objective"] = m.sense * objective;
        report["verified_lower_bound"] =
            std::isfinite(best_bound) ? json(m.sense * best_bound) : json();
        double gap = std::isfinite(best_bound) ? std::max(0., objective - best_bound) : inf;
        report["absolute_gap"] = std::isfinite(gap) ? json(gap) : json();
        report["verification"] = {{"primal_residual", acc.primal},
                                  {"dual_residual", acc.dual},
                                  {"kkt_error", acc.kkt},
                                  {"feasible", acc.finite && acc.primal <= 1e-6}};
        json solution = json::object();
        for (int64_t j = 0; j < n; ++j)
            solution[m.names[j]] = master_x[j];
        report["solution"] = solution;
        if (status == "OPTIMAL" && !(acc.finite && acc.primal <= 1e-6))
            status = "NUMERICAL_ERROR";
    }
    report["status"] = status;
    report["scope"] = std::string("Continuous LP with bordered block-diagonal structure. The "
                                  "bound is the verifier's Lagrangian bound on the original "
                                  "model using master and pricing duals.") +
                      (boxed ? " Unbounded block columns were boxed at +/-1e6*scale, so the "
                               "bound is withheld." : "");
    return report.dump(2);
}
} // namespace vantage
