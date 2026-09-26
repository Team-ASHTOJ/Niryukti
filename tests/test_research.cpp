#include "../src/internal.hpp"
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
void gpu_execution() {
    if (!cuda_available())
        return;
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
