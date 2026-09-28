#include "../src/internal.hpp"
#include <array>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace vantage;
namespace {
int checks = 0;
void require(bool condition, const char *message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}
void near(double a, double b, double tolerance = 1e-11) {
    require(std::abs(a - b) <= tolerance * (1 + std::abs(b)), "numerical disagreement");
}
Model model(int n, int rows, std::vector<Entry> entries) {
    Model m;
    m.A = Sparse::build(rows, n, std::move(entries));
    m.c.assign(n, 0);
    m.q.assign(n, 0);
    m.lb.assign(n, -10);
    m.ub.assign(n, 10);
    m.types.assign(n, VarType::Continuous);
    m.rl.assign(rows, -inf);
    m.ru.assign(rows, inf);
    for (int j = 0; j < n; ++j)
        m.names.push_back("x" + std::to_string(j));
    for (int i = 0; i < rows; ++i)
        m.row_names.push_back("r" + std::to_string(i));
    return m;
}
void acceptance() {
    auto m = model(1, 1, {{0, 0, 1}});
    m.c = {-1};
    m.rl = m.ru = {3};
    // First trial: dx=dy=2, E=4, C=4, limit=1/2. The old CPU rule accepted it.
    auto cpu = cpu_backend(m, {0}, {0}, true);
    require(cpu->advance(1, 2, 2) == 1, "one accepted CPU step");
    require(cpu->rejected_steps() > 0, "unsafe trial must be rejected");
    std::vector<double> x, y, ax, ay;
    cpu->candidates(x, y, ax, ay);
    require(x[0] > 0 && x[0] < 2, "backtracking reduced step");
    // Here the accepted tau equals dx, independently of the line-search implementation.
    require((x[0] * x[0] + y[0] * y[0]) / x[0] >= 2 * std::abs(x[0] * y[0]),
            "accepted energy inequality");
    near(ax[0], x[0]);
    near(ay[0], y[0]);
    if (cuda_available()) {
        auto gpu = cuda_backend(m, {0}, {0}, true);
        int accepted = 0;
        for (int attempt = 0; attempt < 40 && !accepted; ++attempt)
            accepted += gpu->advance(1, 2, 2);
        require(accepted == 1, "one accepted CUDA step");
        require(gpu->rejected_steps() == cpu->rejected_steps(), "backtracking parity");
        std::vector<double> gx, gy, gax, gay;
        gpu->candidates(gx, gy, gax, gay);
        near(gx[0], x[0]);
        near(gy[0], y[0]);
        near(gax[0], ax[0]);
        near(gay[0], ay[0]);
    }
}
void halpern() {
    auto m = model(1, 1, {{0, 0, 1}});
    m.lb = {-inf};
    m.ub = {inf};
    m.rl = m.ru = {1};
    auto backend = cpu_backend(m, {0}, {0}, false, true);
    // Independent scalar recurrence for min_x max_y (x-1)y.
    double expected_x = 0, expected_y = 0;
    for (int k = 0; k < 20; ++k) {
        double tx = expected_x - .5 * expected_y;
        double ty = expected_y + .5 * (2 * tx - expected_x - 1);
        double w = double(k + 1) / (k + 2);
        expected_x = w * tx;
        expected_y = w * ty;
        backend->advance(1, .5, .5);
        std::vector<double> x, y, rawx, rawy;
        backend->candidates(x, y, rawx, rawy);
        near(x[0], expected_x);
        near(y[0], expected_y);
        near(rawx[0], tx);
        near(rawy[0], ty);
    }
    backend->reset({2}, {3});
    backend->advance(1, .5, .5);
    std::vector<double> x, y, rawx, rawy;
    backend->candidates(x, y, rawx, rawy);
    near(x[0], 1.25);
    near(y[0], 2.5);
    // LP with both finite bounds and ranged constraints; optimum x=2, y=0.
    auto lp = model(2, 2, {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, -1}});
    lp.lb = {0, 0};
    lp.ub = {4, 4};
    lp.c = {1, 2};
    lp.rl = {2, -1};
    lp.ru = {3, 2};
    for (auto scaling : {"ruiz", "combined"}) {
        Options o;
        o.method = "halpern";
        o.adaptive = false;
        o.scaling = scaling;
        auto r = solve(lp, o);
        require(r.status == "OPTIMAL", "Halpern LP convergence");
        near(r.accuracy.objective, 2, 1e-5);
        require(verify(lp, r.x, r.y).kkt <= o.tol, "Halpern original-space KKT");
        o.initial_x = r.x;
        o.initial_y = r.y;
        require(solve(lp, o).iterations == 0, "Halpern warm-start round trip");
        o.initial_x.clear();
        o.initial_y.clear();
        o.iteration_limit = 0;
        require(solve(lp, o).status == "ITERATION_LIMIT", "Halpern limits preserved");
    }
}
void sparse_quadratic() {
    auto m = model(2, 1, {{0, 0, 1}, {0, 1, 1}});
    m.lb = {0, 0};
    m.ub = {4, 4};
    m.rl = m.ru = {3};
    m.Q = Sparse::build(2, 2, {{0, 0, 2}, {0, 1, 1}, {1, 0, 1}, {1, 1, 2}});
    m.c = {-4, -5}; // Optimum (1,2), objective -7.
    m.validate();
    for (auto device : {"cpu", "cuda"}) {
        if (std::string(device) == "cuda" && !cuda_available())
            continue;
        Options o;
        o.device = device;
        o.iteration_limit = 100000;
        auto r = solve(m, o);
        require(r.status == "OPTIMAL", "sparse QP convergence");
        near(r.x[0], 1, 1e-5);
        near(r.x[1], 2, 1e-5);
        near(r.accuracy.objective, -7, 1e-5);
        require(verify(m, r.x, r.y).kkt <= o.tol, "sparse QP original KKT");
        if (std::string(device) == "cuda") {
            o.cuda_graphs = true;
            auto graph = solve(m, o);
            require(graph.status == "OPTIMAL", "sparse QP graph solve");
            near(graph.accuracy.objective, r.accuracy.objective, 1e-6);
        }
    }
    auto fingerprint = m.fingerprint();
    write_model(m, "/tmp/vantage-sparse-qp-test.json");
    auto copy = read_model("/tmp/vantage-sparse-qp-test.json");
    require(copy.fingerprint() == fingerprint, "full Q JSON roundtrip");
    write_model(m, "/tmp/vantage-sparse-qp-test.mps");
    copy = read_model("/tmp/vantage-sparse-qp-test.mps");
    auto a = verify(copy, {1, 2}, {0});
    near(a.objective, -7);
    auto bad = m;
    bad.Q = Sparse::build(2, 2, {{0, 0, 1}, {0, 1, 2}, {1, 0, 2}, {1, 1, 1}});
    bool rejected = false;
    try {
        bad.validate();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "indefinite sparse Q rejected");
    bad = m;
    bad.Q.value[1] = .9;
    rejected = false;
    try {
        bad.validate();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "asymmetric sparse Q rejected");
}
void quadratic_extensions() {
    auto m = model(2, 1, {{0, 0, 1}, {0, 1, 1}});
    m.lb = {0, 0};
    m.ub = {3, 3};
    m.rl = {2.5};
    m.ru = {inf};
    m.c = {-2, -3};
    m.Q = Sparse::build(2, 2, {{0, 0, 2}, {0, 1, 1}, {1, 0, 1}, {1, 1, 2}});
    m.types.assign(2, VarType::Integer);
    double oracle = inf;
    for (int x = 0; x <= 3; ++x)
        for (int y = 0; y <= 3; ++y) {
            auto a = verify(m, {double(x), double(y)}, {0});
            if (a.primal == 0)
                oracle = std::min(oracle, a.objective);
        }
    for (auto branching : {"fractional", "reliability"}) {
        Options o;
        o.device = "cpu";
        o.branching = branching;
        o.primal_heuristic = "all";
        o.time_limit = 5;
        auto r = solve(m, o);
        require(r.status == "OPTIMAL", "MIQP branch-and-bound");
        near(r.accuracy.objective, oracle, 1e-5);
        require(r.best_bound <= oracle + 1e-9, "MIQP interval lower bound");
    }
    m.types.assign(2, VarType::Continuous);
    m.lb[0] = m.ub[0] = 1;
    auto original = solve(m, Options{});
    require(original.status == "OPTIMAL", "cross-term fixed variable substitution");
    near(original.x[0], 1);
    require(original.removed_columns > 0, "full Q fixed column removed");
    Options o;
    o.presolve = false;
    o.scaling_passes = 0;
    auto plain = solve(m, o);
    require(plain.status == "OPTIMAL", "unscaled full Q solve");
    near(plain.accuracy.objective, original.accuracy.objective, 1e-5);
    // Every independently evaluated point gives a bound below the exact finite-box optimum.
    m.lb = {0, 0};
    m.ub = {3, 3};
    m.rl = {-inf};
    m.ru = {inf};
    double optimum = -7. / 3.; // unconstrained Qx = -c gives x=1/3,y=4/3.
    for (int k = 0; k < 20; ++k) {
        std::vector<double> x = {double(k) / 7, 3 - double(k) / 7};
        auto a = verify(m, x, {0});
        require(a.lower_bound <= optimum + 1e-12, "QP affine bound sound");
    }
    m.Q = Sparse::build(2, 2, {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}});
    m.c = {-1, -1};
    m.validate();
    require(solve(m).status == "OPTIMAL", "singular PSD QP");
    if (cuda_available()) {
        Options g;
        g.device = "cuda";
        g.cuda_graphs = true;
        g.gpu_monitor = true;
        auto r = solve(m, g);
        require(r.status == "OPTIMAL", "GPU monitored QP");
        require(r.monitor_checks > 0, "device monitor exercised");
        require(verify(m, r.x, r.y).kkt <= g.tol,
                "monitor never substitutes for independent verifier");
    }
}
void sparse_qp_recession_guard() {
    auto m = model(1, 0, {});
    m.lb = {-inf};
    m.ub = {inf};
    m.c = {-2};
    m.Q = Sparse::build(1, 1, {{0, 0, 2}});
    Options o;
    auto r = solve(m, o);
    require(r.status == "OPTIMAL", "sparse quadratic column is not linear unbounded ray");
    near(r.x[0], 1, 1e-6);
    near(r.accuracy.objective, -1, 1e-6);
}
void binary_conflict_propagation() {
    std::vector<std::vector<std::pair<int64_t, int>>> clauses = {{{0, 1}, {1, 1}},
                                                                 {{1, 0}, {2, 1}}};
    std::vector<double> lb = {1, 0, 0}, ub = {1, 1, 1};
    int64_t changes = 0;
    require(propagate_binary_conflicts(clauses, lb, ub, changes),
            "conflict unit propagation feasible");
    require(ub[1] == 0 && ub[2] == 0 && changes == 2, "conflict implications cascade");
    for (int mask = 0; mask < 8; ++mask) {
        bool satisfies = true;
        for (const auto &clause : clauses) {
            bool matches = true;
            for (auto [j, v] : clause)
                matches &= ((mask >> j) & 1) == v;
            satisfies &= !matches;
        }
        if (satisfies && (mask & 1))
            for (int j = 0; j < 3; ++j)
                require(((mask >> j) & 1) >= lb[j] && ((mask >> j) & 1) <= ub[j],
                        "conflicts preserve every feasible binary assignment");
    }
    lb = {1, 1, 0};
    ub = {1, 1, 1};
    require(!propagate_binary_conflicts(clauses, lb, ub, changes), "conflict contradiction");
}
void simplex_and_cuts() {
    std::mt19937 generator(26119);
    for (int trial = 0; trial < 40; ++trial) {
        std::vector<Entry> entries;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 2; ++j)
                entries.push_back({i, j, double(int(generator() % 9) - 4)});
        auto m = model(2, 3, entries);
        m.lb = {0, 0};
        m.ub = {4, 4};
        m.c = {double(int(generator() % 7) - 3), double(int(generator() % 7) - 3)};
        auto activity = m.A.multiply({1, 2});
        for (int i = 0; i < 3; ++i)
            m.ru[i] = activity[i] + .5;
        std::vector<std::array<double, 3>> lines = {
            {{1, 0, 0}}, {{0, 1, 0}}, {{1, 0, 4}}, {{0, 1, 4}}};
        for (int i = 0; i < 3; ++i) {
            double a = 0, b = 0;
            for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
                (m.A.index[k] ? b : a) = m.A.value[k];
            lines.push_back({a, b, m.ru[i]});
        }
        double oracle = inf;
        for (size_t i = 0; i < lines.size(); ++i)
            for (size_t j = i + 1; j < lines.size(); ++j) {
                auto a = lines[i], b = lines[j];
                double determinant = a[0] * b[1] - a[1] * b[0];
                if (std::abs(determinant) < 1e-12)
                    continue;
                std::vector<double> point = {(a[2] * b[1] - a[1] * b[2]) / determinant,
                                             (a[0] * b[2] - a[2] * b[0]) / determinant};
                auto v = verify(m, point, {0, 0, 0});
                if (v.primal <= 1e-9)
                    oracle = std::min(oracle, v.objective);
            }
        Options o;
        o.method = "simplex";
        auto r = solve(m, o);
        require(r.status == "OPTIMAL", "simplex random bounded LP");
        near(r.accuracy.objective, oracle, 1e-7);
        require(r.accuracy.kkt <= o.tol, "simplex independently verified");
    }
    auto m = model(2, 1, {{0, 0, 2}, {0, 1, .5}});
    m.lb = {0, 0};
    m.ub = {3, 5};
    m.ru = {2.5};
    m.types[0] = VarType::Integer;
    auto cuts = m;
    require(add_mir_cuts(cuts, 4) > 0, "MIR generated");
    for (int x = 0; x <= 3; ++x)
        for (int y = 0; y <= 100; ++y) {
            std::vector<double> point = {double(x), double(y) / 20};
            auto before = verify(m, point, {0});
            if (before.primal <= 1e-12)
                require(verify(cuts, point, std::vector<double>(cuts.rl.size())).primal <= 1e-10,
                        "MIR preserves feasible grid");
        }
    // Independent feasible-grid oracle checks MIR with both signs and shifted bounds.
    for (int trial = 0; trial < 80; ++trial) {
        auto base = model(3, 1,
                          {{0, 0, (int(generator() % 17) - 8) / 4.},
                           {0, 1, (int(generator() % 17) - 8) / 4.},
                           {0, 2, (int(generator() % 17) - 8) / 4.}});
        base.lb = {-2, -1, -.5};
        base.ub = {2, 3, 2.5};
        base.types = {VarType::Integer, VarType::Integer, VarType::Continuous};
        double endpoint = (int(generator() % 31) - 15) / 4. + .125;
        if (trial % 2)
            base.rl[0] = endpoint;
        else
            base.ru[0] = endpoint;
        auto tightened = base;
        add_mir_cuts(tightened, 4);
        for (int x = -2; x <= 2; ++x)
            for (int y = -1; y <= 3; ++y)
                for (int z = -2; z <= 10; ++z) {
                    std::vector<double> point = {double(x), double(y), z / 4.};
                    if (verify(base, point, {0}).primal <= 1e-12)
                        require(verify(tightened, point, std::vector<double>(tightened.rl.size()))
                                        .primal <= 1e-10,
                                "MIR preserves independently enumerated mixed feasible points");
                }
    }
    m.c = {-1, 0};
    Options o;
    o.method = "auto";
    o.cuts = true;
    o.branching = "reliability";
    auto r = solve(m, o);
    require(r.status == "OPTIMAL", "simplex-backed branch and cut");
    near(r.accuracy.objective, -1);
    for (const auto &policy : {"best-bound", "depth-first", "best-estimate"}) {
        o.node_selection = policy;
        r = solve(m, o);
        require(r.status == "OPTIMAL", "node-selection policy verified");
        near(r.accuracy.objective, -1);
    }
    auto cycling = model(4, 3,
                         {{0, 0, .5},
                          {0, 1, -5.5},
                          {0, 2, -2.5},
                          {0, 3, 9},
                          {1, 0, .5},
                          {1, 1, -1.5},
                          {1, 2, -.5},
                          {1, 3, 1},
                          {2, 0, 1}});
    cycling.lb = {0, 0, 0, 0};
    cycling.ub = {inf, inf, inf, inf};
    cycling.ru = {0, 0, 1};
    cycling.c = {-10, 57, 9, 24};
    o.method = "simplex";
    o.presolve = false;
    o.scaling_passes = 0;
    auto cycled = solve(cycling, o);
    require(cycled.status == "OPTIMAL", "Bland fallback terminates classical cycling LP");
    near(cycled.accuracy.objective, -1, 1e-7);
    o.presolve = true;
    o.scaling_passes = 5;
    auto bad = model(2, 2, {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}});
    bad.lb = {0, 0};
    bad.ub = {10, 10};
    bad.rl = {3, -inf};
    bad.ru = {inf, 1};
    o.method = "simplex";
    r = solve(bad, o);
    require(r.status == "INFEASIBLE", "simplex Farkas proof");
    require(Verifier(bad).infeasibility_bound(r.infeasibility_ray) > 0,
            "simplex ray independently verified");
}
void completion_regressions() {
    {
        auto duplicate = model(1, 3, {{0, 0, 1}, {1, 0, 1}, {2, 0, 1}});
        duplicate.lb = {0};
        duplicate.ub = {10};
        duplicate.c = {-1};
        duplicate.rl = {2, -inf, -inf};
        duplicate.ru = {inf, 3, 8};
        Options o;
        o.method = "auto";
        o.device = "cpu";
        auto r = solve(duplicate, o);
        require(r.status == "OPTIMAL", "parallel rows restore upper source dual");
        near(r.x[0], 3, 1e-6);
        require(r.removed_rows == 2, "parallel row aggregation reduces model");
        duplicate.c[0] = 1;
        r = solve(duplicate, o);
        require(r.status == "OPTIMAL", "parallel rows restore lower source dual");
        near(r.x[0], 2, 1e-6);
        duplicate.ru[1] = 1;
        require(solve(duplicate, o).status == "INFEASIBLE", "identical-row contradiction proof");
    }
    auto m = model(3, 1, {{0, 0, 2}, {0, 1, 2}, {0, 2, 2}});
    m.lb = {0, 0, 0};
    m.ub = {1, 1, 1};
    m.types.assign(3, VarType::Binary);
    m.c = {-1, -1, -1};
    m.ru = {3};
    Options o;
    o.method = "auto";
    o.device = "cpu";
    o.branching = "reliability";
    auto complete = solve(m, o);
    require(complete.status == "OPTIMAL", "baseline tree solves");
    auto directory = std::filesystem::temp_directory_path() /
                     ("vantage-tree-" + std::to_string(Clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    o.node_limit = 1;
    o.checkpoint_path = (directory / "tree.json").string();
    auto partial = solve(m, o);
    require(partial.nodes_remaining > 0, "checkpoint retains open branches");
    require(std::filesystem::exists(o.checkpoint_path), "tree checkpoint written");
    o.node_limit = 10000;
    o.resume_path = o.checkpoint_path;
    auto resumed = solve(m, o);
    require(resumed.status == "OPTIMAL", "resumed tree solves");
    near(resumed.accuracy.objective, complete.accuracy.objective);
    require(resumed.nodes == complete.nodes, "resume preserves processed tree state");
    auto wrong = m;
    wrong.c[0] = -2;
    bool rejected = false;
    try {
        solve(wrong, o);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "resume rejects changed model");
    o.cuts = true;
    rejected = false;
    try {
        solve(m, o);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "resume rejects changed algorithm configuration");
    std::filesystem::remove_all(directory);
    for (const std::string device : {"cpu", "cuda"}) {
        if (device == "cuda" && !cuda_available())
            continue;
        auto lp = model(2, 1, {{0, 0, 1}, {0, 1, 2}});
        lp.c = {1, 1};
        lp.rl = {4};
        lp.ub = {10, 10};
        Options state;
        state.device = device;
        state.iteration_limit = 10000;
        auto direct = solve(lp, state);
        require(direct.status == "OPTIMAL", "continuous direct solve");
        auto path = std::filesystem::temp_directory_path() /
                    ("vantage-state-" + device +
                     std::to_string(Clock::now().time_since_epoch().count()) + ".json");
        state.iteration_limit = 100;
        state.check_every = 10;
        state.checkpoint_nodes = 20;
        state.checkpoint_path = path.string();
        auto limited = solve(lp, state);
        require(std::filesystem::exists(path), "continuous state snapshot exists");
        state.iteration_limit = 10000;
        state.resume_path = path.string();
        auto resumed = solve(lp, state);
        require(resumed.status == "OPTIMAL", "continuous full-state continuation");
        near(resumed.accuracy.objective, direct.accuracy.objective, 1e-5);
        auto wrong = lp;
        wrong.c[0] = 2;
        bool rejected = false;
        try {
            solve(wrong, state);
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "continuous resume rejects model edit");
        std::filesystem::remove(path);
    }
    for (const std::string heuristic : {"local", "all"}) {
        Options strong;
        strong.device = "cpu";
        strong.method = "auto";
        strong.cuts = true;
        strong.branching = "reliability";
        strong.primal_heuristic = heuristic;
        auto solved = solve(m, strong);
        require(solved.status == "OPTIMAL", "cuts/conflicts/local branching verified optimum");
        near(solved.accuracy.objective, -1, 1e-5);
    }
    {
        auto schedule = read_model("examples/scheduling.json");
        Options o;
        o.device = "cpu";
        o.method = "auto";
        o.branching = "reliability";
        o.cuts = true;
        o.primal_heuristic = "all";
        auto result = solve(schedule, o);
        require(result.status == "OPTIMAL",
                "continuous node bound propagation closes scheduling proof");
        near(result.accuracy.objective, 748, 1e-5);
    }
    // Separation adds only violated cuts and retains every feasible mixed assignment.
    auto mixed = model(2, 1, {{0, 0, 2}, {0, 1, .5}});
    mixed.lb = {0, 0};
    mixed.ub = {3, 5};
    mixed.ru = {2.5};
    mixed.types[0] = VarType::Integer;
    auto separated = mixed;
    std::vector<double> fractional = {1.25, 0};
    require(add_mir_cuts(separated, 4, &fractional) > 0, "violated MIR separated");
    auto satisfied = mixed;
    std::vector<double> integral = {1, 0};
    require(add_mir_cuts(satisfied, 4, &integral) == 0, "nonviolated MIR not appended");
    for (int x = 0; x <= 3; ++x)
        for (int y = 0; y <= 100; ++y) {
            std::vector<double> point = {double(x), y / 20.};
            if (verify(mixed, point, {0}).primal <= 1e-12)
                require(verify(separated, point, std::vector<double>(separated.rl.size())).primal <=
                            1e-10,
                        "separated MIR preserves mixed feasible points");
        }
    // Large non-dominant SPD blocks: optimum is exactly the all-one vector.
    auto qp = model(300, 0, {});
    qp.lb.assign(300, 0);
    qp.ub.assign(300, 2);
    std::vector<Entry> entries;
    for (int i = 0; i < 300; i += 2) {
        entries.insert(entries.end(), {{i, i, 1}, {i, i + 1, 2}, {i + 1, i, 2}, {i + 1, i + 1, 5}});
        qp.c[i] = -3;
        qp.c[i + 1] = -7;
    }
    qp.Q = Sparse::build(300, 300, entries);
    bool supported = true;
    try {
        qp.validate();
    } catch (const std::exception &e) {
        if (std::string(e.what()).find("enable VANTAGE_SPARSE_LU") != std::string::npos)
            supported = false;
        else
            throw;
    }
    {
        auto lp = model(2, 1, {{0, 0, 1}, {0, 1, 1}});
        lp.lb = {0, 0};
        lp.ub = {10, 10};
        lp.ru = {10};
        lp.c = {-2, -1};
        Options dual;
        dual.device = "cpu";
        dual.method = "dual-simplex";
        dual.presolve = false;
        dual.scaling_passes = 0;
        auto cold = solve(lp, dual);
        require(cold.status == "OPTIMAL", "dual engine cold basis initialization");
        require(!cold.basis.empty(), "simplex exports explicit basis");
        lp.ub[0] = 4;
        dual.initial_basis = cold.basis;
        dual.basis_fingerprint = cold.basis_fingerprint;
        auto warm = solve(lp, dual);
        require(warm.status == "OPTIMAL", "dual reoptimization solves changed bound");
        require(warm.method_selected == "revised-dual-simplex", "actual dual pivot path used");
        near(warm.accuracy.objective, -14, 1e-6);
        near(warm.x[0], 4, 1e-6);
        near(warm.x[1], 6, 1e-6);
        dual.initial_basis.assign(cold.basis.size(), -1);
        require(solve(lp, dual).status == "OPTIMAL", "invalid warm basis safely cold starts");
        Options portfolio;
        portfolio.method = "concurrent";
        portfolio.device = "cpu";
        auto raced = solve(lp, portfolio);
        require(raced.status == "OPTIMAL", "concurrent LP portfolio verifies winner");
        near(raced.accuracy.objective, -14, 1e-6);
        portfolio.cancellation = std::make_shared<std::atomic<bool>>(true);
        auto stopped = solve(lp, portfolio);
        require(stopped.status != "NUMERICAL_ERROR", "portfolio cancellation remains safe");
        Options barrier;
        barrier.method = "barrier";
        barrier.device = "cpu";
        auto interior = solve(lp, barrier);
        if (interior.status != "UNSUPPORTED") {
            require(interior.status == "OPTIMAL", "predictor corrector LP barrier");
            near(interior.accuracy.objective, -14, 1e-5);
            auto qp = model(2, 1, {{0, 0, 1}, {0, 1, 1}});
            qp.lb = {0, 0};
            qp.ub = {10, 10};
            qp.rl = qp.ru = {3};
            qp.Q = Sparse::build(2, 2, {{0, 0, 2}, {0, 1, 1}, {1, 0, 1}, {1, 1, 2}});
            auto q = solve(qp, barrier);
            require(q.status == "OPTIMAL", "barrier sparse convex QP with equality");
            near(q.x[0], 1.5, 1e-5);
            near(q.x[1], 1.5, 1e-5);
            near(q.accuracy.objective, 6.75, 1e-5);
            portfolio.cancellation.reset();
            auto qrace = solve(qp, portfolio);
            require(qrace.status == "OPTIMAL", "concurrent sparse QP portfolio");
        }
    }
    if (cuda_available()) {
        {
            auto lp = model(2, 1, {{0, 0, 1}, {0, 1, 1}});
            lp.lb = {0, 0};
            lp.ub = {10, 10};
            lp.ru = {10};
            lp.c = {-2, -1};
            Options gpu;
            gpu.device = "cuda";
            gpu.method = "barrier";
            auto result = solve(lp, gpu);
            require(result.status == "OPTIMAL", "GPU Newton barrier verified LP");
            near(result.accuracy.objective, -20, 1e-5);
            auto qp = model(2, 1, {{0, 0, 1}, {0, 1, 1}});
            qp.lb = {0, 0};
            qp.ub = {10, 10};
            qp.rl = qp.ru = {3};
            qp.Q = Sparse::build(2, 2, {{0, 0, 2}, {0, 1, 1}, {1, 0, 1}, {1, 1, 2}});
            result = solve(qp, gpu);
            require(result.status == "OPTIMAL", "GPU Newton barrier verified QP");
            near(result.accuracy.objective, 6.75, 1e-5);
        }
        // Device-directed bounds must preserve every enumerated feasible integer assignment.
        auto integer =
            model(3, 2, {{0, 0, .25}, {0, 1, -2}, {0, 2, 1}, {1, 0, 1}, {1, 1, 1}, {1, 2, 1}});
        integer.lb = {-2, -2, -2};
        integer.ub = {3, 3, 3};
        integer.types.assign(3, VarType::Integer);
        integer.rl = {.125, 2};
        integer.ru = {2.375, 3};
        auto low = integer.lb, high = integer.ub;
        bool feasible = cuda_propagate_integer_bounds(integer, low, high);
        int count = 0;
        for (int x = -2; x <= 3; ++x)
            for (int y = -2; y <= 3; ++y)
                for (int z = -2; z <= 3; ++z) {
                    std::vector<double> point = {double(x), double(y), double(z)};
                    if (verify(integer, point, {0, 0}).primal <= 1e-12) {
                        count++;
                        require(feasible, "GPU propagation cannot remove feasible domain");
                        for (int j = 0; j < 3; ++j)
                            require(point[j] >= low[j] && point[j] <= high[j],
                                    "GPU bounds preserve feasible assignment");
                    }
                }
        require(count > 0, "GPU propagation oracle has feasible points");
        auto continuous = model(2, 1, {{0, 0, 2}, {0, 1, 1}});
        continuous.lb = {0, 0};
        continuous.ub = {10, 1};
        continuous.rl = {3.5};
        continuous.ru = {4.5};
        auto cl = continuous.lb, cu = continuous.ub;
        require(cuda_propagate_integer_bounds(continuous, cl, cu), "GPU continuous propagation");
        require(cl[0] > 1.2 && cu[0] < 2.3,
                "GPU continuous endpoints tightened without integer rounding");
        for (double x : {1.25, 1.75, 2.25})
            for (double y : {0., .5, 1.})
                if (verify(continuous, {x, y}, {0}).primal == 0)
                    require(x >= cl[0] && x <= cu[0] && y >= cl[1] && y <= cu[1],
                            "GPU continuous propagation preserves boundary feasible points");
        Options presolve_gpu;
        presolve_gpu.device = "cuda";
        presolve_gpu.gpu_presolve = true;
        continuous.c = {1, 0};
        auto continuous_result = solve(continuous, presolve_gpu);
        require(continuous_result.status == "OPTIMAL", "continuous GPU presolve end-to-end");
        near(continuous_result.accuracy.objective, 1.25, 1e-5);
        auto impossible = integer;
        impossible.rl[1] = 10;
        impossible.ru[1] = 11;
        low = impossible.lb;
        high = impossible.ub;
        require(!cuda_propagate_integer_bounds(impossible, low, high),
                "GPU contradiction detection");
        auto lp = model(2, 1, {{0, 0, 1}, {0, 1, 1}});
        lp.lb = {0, 0};
        lp.ub = {5, 5};
        lp.rl = {3};
        lp.c = {1, 2};
        auto resident = cuda_backend(lp, {0, 0}, {0});
        resident->advance(20, .1, .1);
        std::vector<double> rx, ry, ra, rb;
        resident->candidates(rx, ry, ra, rb);
        auto state = resident->snapshot();
        require(resident->reset_candidate(true), "CUDA resident averaged restart supported");
        resident->advance(20, .1, .1);
        std::vector<double> sx, sy, sa, sb;
        resident->candidates(sx, sy, sa, sb);
        resident->restore(state);
        resident->reset(ra, rb);
        resident->advance(20, .1, .1);
        resident->candidates(rx, ry, ra, rb);
        for (size_t j = 0; j < rx.size(); ++j)
            near(rx[j], sx[j], 1e-12);
        for (size_t j = 0; j < ry.size(); ++j)
            near(ry[j], sy[j], 1e-12);
        Options bo;
        bo.device = "cuda";
        bo.iteration_limit = 5000;
        auto batch = cuda_batch_relaxations(lp, {{0, 0}, {0, 1}}, {{2, 5}, {5, 5}}, bo);
        require(batch.size() == 2, "two GPU relaxations returned");
        for (size_t b = 0; b < batch.size(); ++b) {
            auto child = lp;
            child.lb = b ? std::vector<double>{0, 1} : std::vector<double>{0, 0};
            child.ub = b ? std::vector<double>{5, 5} : std::vector<double>{2, 5};
            auto single = solve(child, bo);
            require(batch[b].accuracy.finite, "batched point finite");
            require(batch[b].accuracy.lower_bound <= single.accuracy.objective + 1e-8,
                    "batched safe bound never exceeds optimum");
            near(batch[b].accuracy.objective, 4, 1e-3);
        }
        bo.method = "auto";
        bo.branching = "reliability";
        bo.batch_strong_branching = true;
        bo.gpu_presolve = true;
        auto mip = solve(m, bo);
        require(mip.status == "OPTIMAL", "GPU batched reliability tree");
        near(mip.accuracy.objective, complete.accuracy.objective, 1e-5);
    }
    if (supported) {
        auto singular = model(300, 0, {});
        singular.ub.assign(300, 2);
        std::vector<Entry> entries;
        for (int i = 0; i < 300; i += 2)
            entries.insert(entries.end(),
                           {{i, i, 1}, {i, i + 1, 2}, {i + 1, i, 2}, {i + 1, i + 1, 4}});
        singular.Q = Sparse::build(300, 300, entries);
        singular.validate();
        Options so;
        so.device = "cpu";
        auto answer = solve(singular, so);
        require(answer.status == "OPTIMAL", "large singular non-dominant PSD accepted");
        singular.Q.value.back() -= .001;
        bool negative = false;
        try {
            singular.validate();
        } catch (const std::exception &) {
            negative = true;
        }
        require(negative, "singular PSD checker rejects negative perturbation");
    }
    if (supported) {
        Options qo;
        qo.device = "cpu";
        qo.scaling = "combined";
        auto solved = solve(qp, qo);
        require(solved.status == "OPTIMAL", "large non-dominant SPD QP solves");
        for (double x : solved.x)
            near(x, 1, 1e-5);
        near(solved.accuracy.objective, -750, 1e-6);
        auto indefinite = qp;
        indefinite.q[0] = -1;
        bool rejected = false;
        try {
            indefinite.validate();
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "large indefinite Q rejected");
        if (cuda_available()) {
            qo.device = "cuda";
            qo.gpu_monitor = true;
            qo.cuda_graphs = true;
            auto gpu = solve(qp, qo);
            require(gpu.status == "OPTIMAL", "CUDA large sparse SPD QP");
            near(gpu.accuracy.objective, solved.accuracy.objective, 1e-6);
        }
    }
}
void gpu_execution() {
    if (!cuda_available())
        return;
    {
        auto nonexact = model(2, 2, {{0, 0, .1}, {0, 1, .3}, {1, 0, std::sqrt(2.)}, {1, 1, -.2}});
        nonexact.lb = {0, 0};
        nonexact.ub = {5, 5};
        nonexact.c = {-1, -.7};
        nonexact.ru = {1, 2};
        Options exact;
        exact.device = "cpu";
        auto reference = solve(nonexact, exact);
        require(reference.status == "OPTIMAL", "nonexact matrix reference converges");
        Options mixed;
        mixed.device = "cuda";
        mixed.matrix_precision = "mixed";
        mixed.cuda_graphs = true;
        mixed.gpu_monitor = true;
        auto r = solve(nonexact, mixed);
        require(r.status == "OPTIMAL", "nonexact mixed GPU solve converges");
        require(verify(nonexact, r.x, r.y).kkt <= mixed.tol,
                "FP64 verification controls mixed GPU optimality");
        near(r.accuracy.objective, reference.accuracy.objective, 1e-5);
        mixed.tol = 1e-10;
        mixed.iteration_limit = 1000;
        auto strict = solve(nonexact, mixed);
        require(strict.status != "OPTIMAL" || verify(nonexact, strict.x, strict.y).kkt <= mixed.tol,
                "mixed precision may reach a limit but cannot fake strict optimality");
    }
    auto m = model(2, 2, {{0, 0, 1}, {0, 1, 2}, {1, 0, -1}, {1, 1, 1}});
    m.c = {1, -2};
    m.rl = {-1, -2};
    m.ru = {3, 2};
    for (double reflection : {0., 1.}) {
        for (bool graphs : {false, true}) {
            auto cpu = cpu_backend(m, {0, 0}, {0, 0}, false, true, reflection, false);
            auto gpu = cuda_backend(m, {0, 0}, {0, 0}, false, true, reflection, false, graphs, "32",
                                    "fp64");
            for (int k = 0; k < 30; ++k) {
                require(cpu->advance(1, .1, .1) == gpu->advance(1, .1, .1),
                        "anchored progress parity");
                std::vector<double> x, y, ax, ay, gx, gy, gax, gay;
                cpu->candidates(x, y, ax, ay);
                gpu->candidates(gx, gy, gax, gay);
                for (int j = 0; j < 2; ++j) {
                    near(x[j], gx[j]);
                    near(y[j], gy[j]);
                    near(ax[j], gax[j]);
                    near(ay[j], gay[j]);
                }
                if (k == 14) {
                    cpu->reset(x, y);
                    gpu->reset(gx, gy);
                }
            }
        }
    }
    for (auto width : {"32", "64"})
        for (auto precision : {"fp64", "mixed"}) {
            auto cpu = cpu_backend(m, {0, 0}, {0, 0}, false);
            auto gpu =
                cuda_backend(m, {0, 0}, {0, 0}, false, false, 0, false, true, width, precision);
            cpu->advance(30, .1, .1);
            gpu->advance(30, .1, .1);
            require(std::isfinite(gpu->monitor()), "device scalar diagnostics finite");
            std::vector<double> x, y, ax, ay, gx, gy, gax, gay;
            cpu->candidates(x, y, ax, ay);
            gpu->candidates(gx, gy, gax, gay);
            for (int j = 0; j < 2; ++j) {
                near(x[j], gx[j]);
                near(y[j], gy[j]);
            }
        }
}
void scaling() {
    auto m = model(4, 3, {{0, 0, 1e-4}, {0, 1, -2e3}, {1, 1, 7}, {1, 2, 3e-2}});
    m.c = {3, -2, 7, 1};
    m.q = {2, 3, 0, 4};
    m.offset = 11;
    m.rl = {-3, 2, -inf};
    m.ru = {5, 2, inf};
    m.lb[3] = -inf;
    m.ub[3] = inf;
    Options o;
    o.presolve = false;
    o.scaling = "combined";
    auto p = prepare(m, o);
    std::vector<double> sx = {.2, -.7, 1.3, 2}, sy = {.4, -.6, 0};
    auto x = p.restore_x(sx), y = p.restore_y(sy, 3);
    auto original_ax = m.A.multiply(x), scaled_ax = p.model.A.multiply(sx);
    auto original_aty = m.A.transpose().multiply(y);
    auto scaled_aty = p.model.A.transpose().multiply(sy);
    double f = m.offset, sf = p.model.offset;
    for (size_t j = 0; j < x.size(); ++j) {
        f += m.c[j] * x[j] + .5 * m.q[j] * x[j] * x[j];
        sf += p.model.c[j] * sx[j] + .5 * p.model.q[j] * sx[j] * sx[j];
        near(original_aty[j] * p.column_scale[j], scaled_aty[j] * p.objective_scale);
        if (std::isfinite(m.lb[j]))
            near(p.model.lb[j] * p.column_scale[j], m.lb[j]);
    }
    near(sf * p.objective_scale, f);
    for (size_t i = 0; i < sy.size(); ++i) {
        near(scaled_ax[i], original_ax[i] * p.row_scale[i]);
        if (std::isfinite(m.rl[i]))
            near(p.model.rl[i] / p.row_scale[i], m.rl[i]);
    }
    o.scaling_passes = 0;
    auto unscaled = prepare(m, o);
    require(unscaled.model.fingerprint() == m.fingerprint(), "zero passes disables all scaling");
    // Fixed-variable elimination and primal/dual postsolve of a diagonal QP.
    auto qp = model(2, 1, {{0, 0, 1e4}, {0, 1, 2e4}});
    qp.lb = {1, 0};
    qp.ub = {1, 10};
    qp.q = {2, 2};
    qp.rl = qp.ru = {5e4};
    o.presolve = true;
    o.scaling_passes = 5;
    auto r = solve(qp, o);
    require(r.status == "OPTIMAL", "combined scaling QP postsolve");
    near(r.accuracy.objective, 5, 1e-5);
    require(verify(qp, r.x, r.y).kkt <= o.tol, "combined scaling QP KKT");
    // Random binary problems checked against exhaustive enumeration.
    std::mt19937 rng(42);
    int64_t probes = 0;
    for (int instance = 0; instance < 12; ++instance) {
        std::vector<Entry> entries;
        auto mip = model(5, 1, {});
        mip.lb.assign(5, 0);
        mip.ub.assign(5, 1);
        mip.types.assign(5, VarType::Binary);
        mip.ru = {9};
        for (int j = 0; j < 5; ++j) {
            entries.push_back({0, j, double(1 + rng() % 7)});
            mip.c[j] = -double(1 + rng() % 11);
        }
        mip.A = Sparse::build(1, 5, entries);
        double best = 0;
        for (int mask = 0; mask < 32; ++mask) {
            double cost = 0, weight = 0;
            for (int j = 0; j < 5; ++j)
                if (mask & (1 << j)) {
                    cost += mip.c[j];
                    weight += entries[j].value;
                }
            if (weight <= 9)
                best = std::min(best, cost);
        }
        auto result = solve(mip, o);
        require(result.status == "OPTIMAL", "combined scaling MILP status");
        near(result.accuracy.objective, best, 1e-6);
        require(result.best_bound <= best + 1e-6, "MILP bound remains conservative");
        o.branching = "reliability";
        auto reliability = solve(mip, o);
        require(reliability.status == "OPTIMAL", "reliability MILP status");
        near(reliability.accuracy.objective, best, 1e-6);
        require(reliability.best_bound <= best + 1e-6, "pseudocosts never become proof bounds");
        probes += reliability.strong_branch_probes;
        o.branching = "fractional";
    }
    require(probes > 0, "strong branching probes exercised");
    auto mip = model(2, 1, {{0, 0, 2}, {0, 1, 2}});
    mip.lb = {0, 0};
    mip.ub = {1, 1};
    mip.c = {-1, -1};
    mip.ru = {3};
    mip.types.assign(2, VarType::Binary);
    o.branching = "reliability";
    o.node_limit = 1;
    o.mip_gap = 0;
    auto limited = solve(mip, o);
    require(limited.status == "NODE_LIMIT" && limited.nodes_remaining > 0,
            "probing cannot close the parent or discard its children");
    require(limited.best_bound <= -1, "limited tree bound remains conservative");
    o.time_limit = 0;
    require(solve(mip, o).status == "TIME_LIMIT", "reliability time limit");
}
void enhanced_first_order() {
    near(power_norm(Sparse::build(2, 2, {{0, 0, 3}, {1, 1, 4}}), 100), 4);
    near(power_norm(Sparse::build(1, 2, {{0, 0, 3}, {0, 1, -4}}), 30), 5);
    near(power_norm(Sparse::build(2, 2, {}), 30), 0);
    near(power_norm(Sparse::build(1, 1, {{0, 0, 1e200}}), 10) / 1e200, 1);
    PrimalWeightController pid;
    double w = pid.update(1, 100, 1);
    require(w < 1 && w > 0, "PID displacement feedback direction");
    near(pid.update(w, 0, 1), w);
    for (int k = 0; k < 1000; ++k)
        w = pid.update(w, 1e20L, 1);
    require(std::isfinite(w) && w >= 1e-4 * (1 - 1e-12), "PID saturation and anti-windup");
    auto m = model(1, 1, {{0, 0, 1}});
    m.lb = {-inf};
    m.ub = {inf};
    m.rl = m.ru = {1};
    auto reflected = cpu_backend(m, {0}, {0}, false, true, 1);
    double x = 0, y = 0;
    for (int k = 0; k < 20; ++k) {
        double tx = x - .5 * y, ty = y + .5 * (2 * tx - x - 1);
        double weight = double(k + 1) / (k + 2);
        x = weight * (2 * tx - x);
        y = weight * (2 * ty - y);
        reflected->advance(1, .5, .5);
        std::vector<double> cx, cy, rawx, rawy;
        reflected->candidates(cx, cy, rawx, rawy);
        near(cx[0], x);
        near(cy[0], y);
        near(rawx[0], tx);
        near(rawy[0], ty);
    }
    auto restarting = cpu_backend(m, {0}, {0}, false, true, 1, true);
    require(restarting->advance(100, .5, .5) < 100 && restarting->restart_requested(),
            "fixed-point restart interrupts a chunk");
    restarting->reset({0}, {0});
    require(!restarting->restart_requested(), "restart state cleared");
    auto lp = model(2, 1, {{0, 0, 1}, {0, 1, 2}});
    lp.lb = {0, 0};
    lp.c = {1, 1};
    lp.rl = {4};
    for (auto method : {"rhpdhg", "r2hpdhg"})
        for (auto weighting : {"displacement", "pid"}) {
            Options o;
            o.method = method;
            o.primal_weight = weighting;
            o.adaptive = false;
            auto r = solve(lp, o);
            require(r.status == "OPTIMAL", "restarted Halpern convergence");
            near(r.accuracy.objective, 2, 1e-5);
            require(verify(lp, r.x, r.y).kkt <= o.tol, "restarted Halpern verification");
            o.iteration_limit = 7;
            require(solve(lp, o).iterations <= 7, "short chunk iteration budget");
        }
    Options power;
    power.power_iterations = 30;
    power.primal_weight = "pid";
    require(solve(lp, power).status == "OPTIMAL", "power/PID adaptive solve");
    // Independently known dual-feasibility cones for lower-only, upper-only,
    // free and doubly bounded columns and rows.
    auto cones = model(4, 4, {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}});
    cones.lb = {0, -inf, -inf, 0};
    cones.ub = {inf, 1, inf, 1};
    cones.c = {1, -1, 2, 3};
    cones.rl = {0, -inf, -inf, 0};
    cones.ru = {inf, 1, inf, 1};
    auto dual = dual_feasibility_model(cones);
    require(dual.lb == std::vector<double>({-inf, 0, 0, -inf}) &&
                dual.ub == std::vector<double>({0, inf, 0, inf}),
            "dual multiplier cone signs");
    require(dual.rl == std::vector<double>({-1, -inf, -2, -inf}) &&
                dual.ru == std::vector<double>({inf, 1, -2, inf}),
            "reduced-cost cone signs");
    auto slow = model(2, 2, {{0, 0, 1}, {1, 1, 1e-3}});
    slow.lb = {0, 0};
    slow.ub = {inf, inf};
    slow.c = {1, 1e-3};
    slow.rl = slow.ru = {1, 1e-3};
    Options o;
    o.polishing = true;
    o.scaling_passes = 0;
    o.tol = 1e-10;
    o.iteration_limit = 2000;
    o.initial_x = {.999, .999};
    o.initial_y = {-1, -1};
    auto polished = solve(slow, o);
    require(polished.polishing_attempts > 0, "polishing path exercised");
    require(polished.iterations <= o.iteration_limit &&
                polished.polishing_iterations <= polished.iterations,
            "shared polishing budget");
    if (polished.status == "OPTIMAL")
        require(verify(slow, polished.x, polished.y).kkt <= o.tol,
                "polishing original-space check");
}
void mip_research() {
    // Enumerate every binary assignment to independently check cut validity.
    std::mt19937 gen(194);
    for (int sample = 0; sample < 40; ++sample) {
        std::vector<Entry> entries;
        for (int j = 0; j < 6; ++j)
            entries.push_back({0, j, double(1 + gen() % 12)});
        auto original = model(6, 1, entries);
        original.lb.assign(6, 0);
        original.ub.assign(6, 1);
        original.types.assign(6, VarType::Binary);
        original.ru = {double(1 + gen() % 30)};
        auto cut = original;
        int added = add_binary_cuts(cut, 8);
        require(added >= 0 && added <= 8, "cut limit");
        for (int mask = 0; mask < 64; ++mask) {
            std::vector<double> x(6), before, after;
            for (int j = 0; j < 6; ++j)
                x[j] = (mask >> j) & 1;
            before = original.A.multiply(x);
            after = cut.A.multiply(x);
            if (before[0] <= original.ru[0])
                for (size_t i = 1; i < after.size(); ++i)
                    require(after[i] <= cut.ru[i], "cuts preserve every feasible assignment");
        }
    }
    auto m = model(2, 1, {{0, 0, 2}, {0, 1, 2}});
    m.lb = {0, 0};
    m.ub = {1, 1};
    m.types.assign(2, VarType::Binary);
    m.c = {-1, -1};
    m.ru = {3};
    auto projection = distance_projection_model(m, {1, 1});
    require(!projection.is_mip() && projection.c.size() == 4, "continuous distance formulation");
    auto pr = solve(projection, Options{});
    require(pr.status == "OPTIMAL", "distance projection solved");
    near(pr.accuracy.objective, .5, 1e-5);
    for (auto heuristic : {"repair", "pump", "rins", "all"}) {
        Options o;
        o.cuts = true;
        o.primal_heuristic = heuristic;
        o.tol = 1e-6;
        auto r = solve(m, o);
        require(r.status == "OPTIMAL", "cut and heuristic MILP solves");
        near(r.accuracy.objective, -1, 1e-5);
        require(r.cuts_added > 0, "cut generation exercised");
        require(verify(m, r.x, r.y).primal <= o.tol, "original incumbent feasible");
        o.node_limit = 1;
        r = solve(m, o);
        require(r.nodes <= 1, "heuristic shared node budget");
    }
    auto h = model(3, 1, {{0, 0, 2}, {0, 1, 2}});
    h.lb = {0, 0, 0};
    h.ub = {1, 1, 1};
    h.types.assign(3, VarType::Binary);
    h.c = {-1, -1, -1};
    h.ru = {2.5};
    Options o;
    o.primal_heuristic = "all";
    auto r = solve(h, o);
    require(r.status == "OPTIMAL" && r.pump_rounds > 0 && r.rins_calls > 0 && r.heuristic_nodes > 0,
            "pump and RINS paths exercised");
    near(r.accuracy.objective, -2, 1e-5);
    require(r.best_bound <= -2 + 1e-8, "neighborhood bound does not prune original problem");
    for (int limit : {1, 2, 3}) {
        o.node_limit = limit;
        r = solve(h, o);
        require(r.nodes <= limit, "RINS shares global node limit");
        if (r.status == "OPTIMAL")
            near(r.accuracy.objective, -2, 1e-5);
    }
}
} // namespace
int main() {
    try {
        acceptance();
        halpern();
        gpu_execution();
        sparse_quadratic();
        quadratic_extensions();
        sparse_qp_recession_guard();
        binary_conflict_propagation();
        simplex_and_cuts();
        completion_regressions();
        scaling();
        enhanced_first_order();
        mip_research();
        std::cout << "Passed " << checks << " research assertions; CUDA "
                  << (cuda_available() ? "tested" : "unavailable, skipped") << '\n';
    } catch (const std::exception &e) {
        std::cerr << "FAIL after " << checks << " assertions: " << e.what() << '\n';
        return 1;
    }
}
