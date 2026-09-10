#include "vantage/vantage.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace vantage;
namespace {
int checks = 0;
void require(bool b, const std::string &msg) {
    checks++;
    if (!b)
        throw std::runtime_error(msg);
}
Model make(int n, int rows, std::vector<Entry> e) {
    Model m;
    m.c.assign(n, 0);
    m.q.assign(n, 0);
    m.lb.assign(n, 0);
    m.ub.assign(n, 10);
    m.types.assign(n, VarType::Continuous);
    m.rl.assign(rows, -inf);
    m.ru.assign(rows, inf);
    for (int j = 0; j < n; j++)
        m.names.push_back("x" + std::to_string(j));
    for (int i = 0; i < rows; i++)
        m.row_names.push_back("r" + std::to_string(i));
    m.A = Sparse::build(rows, n, std::move(e));
    return m;
}
void optimum(const Model &m, double obj, Options o = {}) {
    auto r = solve(m, o);
    require(r.status == "OPTIMAL", "status " + r.status + " for " + m.name);
    require(std::abs(r.accuracy.objective - obj) < 1e-4 * (1 + std::abs(obj)),
            "objective disagreement");
    require(r.accuracy.finite && r.accuracy.primal <= o.tol, "primal verification");
    if (!m.is_mip())
        require(r.accuracy.kkt <= o.tol, "KKT verification");
}
} // namespace
int main() {
    try {
        auto m = make(2, 1, {{0, 0, 1}, {0, 1, 2}});
        m.c = {1, 1};
        m.rl = {4};
        optimum(m, 2);
        auto sparse = Sparse::build(2, 2, {{0, 0, 2}, {0, 0, -1}, {1, 1, 3}});
        require(sparse.multiply({2, 4}) == std::vector<double>({2, 12}), "duplicate COO reduction");
        require(sparse.transpose().multiply({2, 4}) == std::vector<double>({2, 12}), "transpose");
        auto fixed = m;
        fixed.lb[0] = fixed.ub[0] = 1;
        optimum(fixed, 2.5);
        Options nop;
        nop.presolve = false;
        optimum(fixed, 2.5, nop);
        auto scaled = m;
        scaled.A.value = {1e8, 2e8};
        scaled.rl = {4e8};
        optimum(scaled, 2);
        auto qp = make(2, 1, {{0, 0, 1}, {0, 1, 1}});
        qp.q = {2, 2};
        qp.rl = qp.ru = {4};
        optimum(qp, 8);
        auto free = make(1, 1, {{0, 0, 1}});
        free.lb = {-inf};
        free.ub = {inf};
        free.c = {1};
        free.rl = free.ru = {-2};
        optimum(free, -2);
        auto infeasible = make(1, 1, {});
        infeasible.rl = {1};
        require(solve(infeasible).status == "INFEASIBLE", "empty row infeasibility");
        auto unbounded = make(1, 0, {});
        unbounded.c = {-1};
        unbounded.ub = {inf};
        require(solve(unbounded).status == "UNBOUNDED", "direct recession certificate");
        auto concave = qp;
        concave.q[0] = -1;
        require(solve(concave).status == "UNSUPPORTED", "nonconvex rejection");
        Options limit;
        limit.iteration_limit = 0;
        require(solve(m, limit).status == "ITERATION_LIMIT", "iteration limit is not optimal");
        limit.time_limit = 0;
        require(solve(m, limit).status == "TIME_LIMIT", "time limit");
        require(!verify(m, {std::numeric_limits<double>::quiet_NaN(), 0}, {0}).finite,
                "NaN candidate rejection");
        require(verify(m, {0, 0}, {0}).primal > 0, "infeasible candidate rejection");
        auto overflow = make(2, 1, {{0, 0, 1e308}, {0, 1, -1e308}});
        require(!verify(overflow, {10, 10}, {0}).finite, "overflow cancellation must not verify");
        auto emptycolumn = make(2, 0, {});
        emptycolumn.c = {1, -1};
        optimum(emptycolumn, -10);
        auto a = verify(m, {0, 2}, {-.6});
        require(a.lower_bound <= 2, "dual bound weak duality");
        auto temp = std::filesystem::temp_directory_path() / "vantage-tests";
        std::filesystem::create_directories(temp);
        for (auto model : {m, qp, free, fixed})
            for (auto ext : {".mps", ".json"}) {
                auto path = (temp / (std::string("roundtrip") + ext)).string();
                write_model(model, path);
                auto read = read_model(path);
                require(read.fingerprint() == model.fingerprint(), "roundtrip " + std::string(ext));
            }
        {
            std::ofstream f(temp / "marker.mps");
            f << "NAME MARKER\nROWS\n N OBJ\nCOLUMNS\n M1 'MARKER' 'INTORG'\n x OBJ -1\n M2 "
                 "'MARKER' 'INTEND'\nENDATA\n";
        }
        auto marker = read_model((temp / "marker.mps").string());
        require(marker.ub[0] == 1, "MPS INTORG default upper bound");
        optimum(marker, -1);
        {
            std::ofstream f(temp / "marker_bound.mps");
            f << "NAME MARKER\nROWS\n N OBJ\nCOLUMNS\n M1 'MARKER' 'INTORG'\n x OBJ 1\n M2 "
                 "'MARKER' 'INTEND'\nBOUNDS\n LO B x 2\nENDATA\n";
        }
        auto marker_bound = read_model((temp / "marker_bound.mps").string());
        require(marker_bound.ub[0] == inf, "MPS explicit bound overrides marker defaults");
        optimum(marker_bound, 2);
        auto toy = read_model("examples/toy.lp");
        optimum(toy, 2);
        {
            std::ofstream f(temp / "syntax.lp");
            f << "Maximize\n obj: 2 x + 3 y + 7\nSubject To\n c: x + y <= 1\nBounds\n 0 <= x <= "
                 "1\n 0 <= y <= 1\nBinary\n x y\nEnd\n";
        }
        auto maximum = read_model((temp / "syntax.lp").string());
        optimum(maximum, -10);
        {
            std::ofstream f(temp / "range.mps");
            f << "NAME RANGE\nROWS\n N OBJ\n E EQ\n L UP\n G LOW\nCOLUMNS\n x OBJ 1 EQ 1\n x UP 1 "
                 "LOW 1\nRHS\n RHS EQ 3 UP 5\n RHS LOW 1\nRANGES\n RNG EQ -2 UP 3\n RNG LOW "
                 "4\nBOUNDS\n FR B x\nENDATA\n";
        }
        auto range = read_model((temp / "range.mps").string());
        require(range.rl == std::vector<double>({1, 2, 1}) &&
                    range.ru == std::vector<double>({3, 5, 5}),
                "MPS ranges");
        optimum(range, 2);
        {
            std::ofstream f(temp / "bad.lp");
            f << "Minimize\n x * y\nSubject To\n c: x >= 1\nEnd\n";
        }
        bool rejected = false;
        try {
            read_model((temp / "bad.lp").string());
        } catch (...) {
            rejected = true;
        }
        require(rejected, "nonlinear LP rejection");
        std::mt19937 rng(7);
        for (int instance = 0; instance < 25; instance++) {
            int n = 5;
            std::vector<Entry> entries;
            std::vector<double> cost(n), weight(n);
            for (int j = 0; j < n; j++) {
                cost[j] = -double(1 + rng() % 12);
                weight[j] = 1 + rng() % 8;
                entries.push_back({0, j, weight[j]});
            }
            auto mip = make(n, 1, entries);
            mip.name = "binary_" + std::to_string(instance);
            mip.c = cost;
            mip.ub.assign(n, 1);
            mip.types.assign(n, VarType::Binary);
            mip.ru = {double(4 + rng() % 12)};
            double best = inf;
            for (int mask = 0; mask < (1 << n); mask++) {
                double w = 0, v = 0;
                for (int j = 0; j < n; j++)
                    if (mask & (1 << j)) {
                        w += weight[j];
                        v += cost[j];
                    }
                if (w <= mip.ru[0])
                    best = std::min(best, v);
            }
            optimum(mip, best);
        }
        if (cuda_available()) {
            Options gpu;
            gpu.device = "cuda";
            optimum(m, 2, gpu);
            optimum(qp, 8, gpu);
            optimum(fixed, 2.5, gpu);
            optimum(free, -2, gpu);
            auto cpu = solve(m), cuda = solve(m, gpu);
            require(std::abs(cpu.accuracy.objective - cuda.accuracy.objective) < 1e-8,
                    "CPU CUDA agreement");
        }
        std::cout << "Passed " << checks << " assertions (including 25 enumerated MILPs); CUDA "
                  << (cuda_available() ? "tested" : "unavailable, skipped") << '\n';
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "FAIL after " << checks << " assertions: " << ex.what() << '\n';
        return 1;
    }
}
