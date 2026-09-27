#include "internal.hpp"
#include <numeric>
#include <stdexcept>
#ifdef VANTAGE_SPARSE_LU
#include <Eigen/SparseLU>
#endif
namespace vantage {
namespace {
// Small-basis numerical kernel only: the model and pricing columns stay sparse.
// PB=LU, partial pivoting, with periodic refactorization and product-form updates.
#ifdef VANTAGE_SPARSE_LU
struct Factor {
    using Matrix = Eigen::SparseMatrix<double>;
    int n;
    bool ready = false;
    mutable Eigen::SparseLU<Matrix> lu;
    explicit Factor(int rows) : n(rows) {}
    void factor(const Sparse &columns, const std::vector<int> &basis) {
        if (!n)
            return;
        std::vector<Eigen::Triplet<double>> entries;
        for (int j = 0; j < n; ++j)
            for (auto k = columns.ptr[basis[j]]; k < columns.ptr[basis[j] + 1]; ++k)
                entries.emplace_back(int(columns.index[k]), j, columns.value[k]);
        Matrix matrix(n, n);
        matrix.setFromTriplets(entries.begin(), entries.end());
        matrix.makeCompressed();
        lu.compute(matrix);
        if (lu.info() != Eigen::Success)
            throw std::runtime_error("Sparse simplex basis factorization failed");
        ready = true;
    }
    std::vector<double> solve(const std::vector<double> &rhs, bool transpose = false) const {
        if (!ready || !n)
            return rhs;
        Eigen::Map<const Eigen::VectorXd> b(rhs.data(), n);
        Eigen::VectorXd x;
        if (transpose)
            x = lu.transpose().solve(b);
        else
            x = lu.solve(b);
        if (!x.allFinite())
            throw std::runtime_error("Nonfinite sparse simplex basis solve");
        return std::vector<double>(x.data(), x.data() + n);
    }
};
#else
struct Factor {
    int n;
    std::vector<double> lu;
    std::vector<int> perm;
    Factor(int n) : n(n), lu(size_t(n) * n), perm(n) {
        std::iota(perm.begin(), perm.end(), 0);
        for (int i = 0; i < n; ++i)
            lu[size_t(i) * n + i] = 1;
    }
    void factor(const Sparse &columns, const std::vector<int> &basis) {
        std::fill(lu.begin(), lu.end(), 0);
        std::iota(perm.begin(), perm.end(), 0);
        for (int j = 0; j < n; ++j)
            for (auto k = columns.ptr[basis[j]]; k < columns.ptr[basis[j] + 1]; ++k)
                lu[size_t(columns.index[k]) * n + j] = columns.value[k];
        for (int k = 0; k < n; ++k) {
            int pivot = k;
            for (int i = k + 1; i < n; ++i)
                if (std::abs(lu[size_t(i) * n + k]) > std::abs(lu[size_t(pivot) * n + k]))
                    pivot = i;
            if (!std::isfinite(lu[size_t(pivot) * n + k]) ||
                std::abs(lu[size_t(pivot) * n + k]) < 1e-14)
                throw std::runtime_error("Singular simplex basis");
            if (pivot != k) {
                for (int j = 0; j < n; ++j)
                    std::swap(lu[size_t(k) * n + j], lu[size_t(pivot) * n + j]);
                std::swap(perm[k], perm[pivot]);
            }
            for (int i = k + 1; i < n; ++i) {
                auto v = lu[size_t(i) * n + k] / lu[size_t(k) * n + k];
                lu[size_t(i) * n + k] = v;
                for (int j = k + 1; j < n; ++j)
                    lu[size_t(i) * n + j] -= v * lu[size_t(k) * n + j];
            }
        }
    }
    std::vector<double> solve(const std::vector<double> &b, bool transpose = false) const {
        std::vector<double> x(n), out(n);
        if (!transpose) {
            for (int i = 0; i < n; ++i) {
                x[i] = b[perm[i]];
                for (int j = 0; j < i; ++j)
                    x[i] -= lu[size_t(i) * n + j] * x[j];
            }
            for (int i = n; i-- > 0;) {
                for (int j = i + 1; j < n; ++j)
                    x[i] -= lu[size_t(i) * n + j] * x[j];
                x[i] /= lu[size_t(i) * n + i];
            }
            return x;
        }
        for (int i = 0; i < n; ++i) {
            x[i] = b[i];
            for (int j = 0; j < i; ++j)
                x[i] -= lu[size_t(j) * n + i] * x[j];
            x[i] /= lu[size_t(i) * n + i];
        }
        for (int i = n; i-- > 0;)
            for (int j = i + 1; j < n; ++j)
                x[i] -= lu[size_t(j) * n + i] * x[j];
        for (int i = 0; i < n; ++i)
            out[perm[i]] = x[i];
        return out;
    }
};
#endif
int64_t basis_row_limit() {
#ifdef VANTAGE_SPARSE_LU
    return 4096;
#else
    return 512;
#endif
}
struct Eta {
    int row;
    std::vector<double> d;
};
struct Standard {
    Sparse columns;
    std::vector<double> b, c, shift, sign;
    std::vector<int> basis, original, row_map;
    std::vector<double> row_sign;
    std::vector<bool> artificial;
};
Standard standardize(const Model &m) {
    Standard s;
    s.shift.resize(m.c.size());
    std::vector<std::vector<int>> map(m.c.size());
    for (size_t j = 0; j < m.c.size(); ++j) {
        auto add = [&](double sign) {
            map[j].push_back(s.c.size());
            s.c.push_back(sign * m.c[j]);
            s.sign.push_back(sign);
            s.original.push_back(j);
        };
        if (std::isfinite(m.lb[j])) {
            s.shift[j] = m.lb[j];
            add(1);
        } else if (std::isfinite(m.ub[j])) {
            s.shift[j] = m.ub[j];
            add(-1);
        } else {
            add(1);
            add(-1);
        }
    }
    std::vector<Entry> e;
    std::vector<int> arts;
    auto row = [&](std::vector<std::pair<int, double>> coefficients, double rhs, int kind,
                   int original, double sign) {
        if (!std::isfinite(rhs))
            throw std::runtime_error("Nonrepresentable simplex transformation");
        if (rhs < 0) {
            rhs = -rhs;
            kind = -kind;
            sign = -sign;
            for (auto &v : coefficients)
                v.second = -v.second;
        }
        int i = s.b.size();
        s.b.push_back(rhs);
        s.row_map.push_back(original);
        s.row_sign.push_back(sign);
        for (auto [j, v] : coefficients)
            e.push_back({i, j, v});
        int slack = -1;
        if (kind != 0) {
            slack = s.c.size();
            s.c.push_back(0);
            s.sign.push_back(0);
            s.original.push_back(-1);
            e.push_back({i, slack, kind < 0 ? 1. : -1.});
        }
        if (kind >= 0) {
            int art = s.c.size();
            s.c.push_back(0);
            s.sign.push_back(0);
            s.original.push_back(-1);
            e.push_back({i, art, 1});
            s.basis.push_back(art);
            arts.push_back(art);
        } else
            s.basis.push_back(slack);
        if (int64_t(s.b.size()) > basis_row_limit())
            throw std::runtime_error("Simplex transformed-row safety limit exceeded; use PDHG");
    };
    for (int64_t i = 0; i < m.A.rows; ++i) {
        std::vector<std::pair<int, double>> coefficients;
        long double offset = 0;
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k) {
            auto j = m.A.index[k];
            offset += (long double)m.A.value[k] * s.shift[j];
            for (auto column : map[j])
                coefficients.push_back({column, m.A.value[k] * s.sign[column]});
        }
        if (m.rl[i] == m.ru[i])
            row(coefficients, m.ru[i] - double(offset), 0, i, 1);
        else {
            if (std::isfinite(m.ru[i]))
                row(coefficients, m.ru[i] - double(offset), -1, i, 1);
            if (std::isfinite(m.rl[i]))
                row(coefficients, m.rl[i] - double(offset), 1, i, 1);
        }
    }
    for (size_t j = 0; j < m.c.size(); ++j)
        if (std::isfinite(m.lb[j]) && std::isfinite(m.ub[j]))
            row({{map[j][0], 1}}, m.ub[j] - m.lb[j], -1, -1, 1);
    s.columns = Sparse::build(s.b.size(), s.c.size(), std::move(e)).transpose();
    s.artificial.assign(s.c.size(), false);
    for (auto j : arts)
        s.artificial[j] = true;
    return s;
}
} // namespace
int64_t simplex_row_limit() {
    return basis_row_limit();
}
static Result solve_simplex_raw(const Model &m, const Options &o) {
    auto start = Clock::now();
    Result r;
    struct Timer {
        Result &r;
        Clock::time_point start;
        ~Timer() {
            r.seconds = elapsed(start);
            r.iteration_seconds =
                std::max(0., r.seconds - r.preprocess_seconds - r.verification_seconds);
        }
    } timer{r, start};
    r.method_selected = "revised-primal-simplex";
    r.device_reason = o.method == "auto" ? "Compact CPU LP selected by automatic method policy"
                                         : "Explicit CPU simplex request";
    r.backend = "cpu";
    r.device_name = "CPU";
    r.status = "UNKNOWN";
#ifdef VANTAGE_SPARSE_LU
    r.message = "Revised primal simplex; own pivot/search logic with Eigen sparse LU numerical "
                "basis kernel";
#else
    r.message = "Revised primal simplex; sparse pricing, small dense basis factorization";
#endif
    if (m.is_qp() || o.device == "cuda") {
        r.status = "UNSUPPORTED";
        r.message = "CPU simplex supports LP relaxations only";
        return r;
    }
    Standard s;
    try {
        s = standardize(m);
    } catch (const std::exception &e) {
        r.status = "UNSUPPORTED";
        r.message = e.what();
        return r;
    }
    r.preprocess_seconds = elapsed(start);
    int rows = s.b.size(), cols = s.c.size();
    Factor factor(rows);
    std::vector<Eta> etas;
    auto solveB = [&](const std::vector<double> &rhs, bool transpose = false) {
        auto v = rhs;
        if (transpose) {
            for (auto k = etas.rbegin(); k != etas.rend(); ++k) {
                double z = v[k->row];
                for (int i = 0; i < rows; ++i)
                    if (i != k->row)
                        z -= k->d[i] * v[i];
                v[k->row] = z / k->d[k->row];
            }
            return factor.solve(v, true);
        }
        v = factor.solve(v);
        for (const auto &e : etas) {
            double z = v[e.row] / e.d[e.row];
            for (int i = 0; i < rows; ++i)
                if (i != e.row)
                    v[i] -= e.d[i] * z;
            v[e.row] = z;
        }
        return v;
    };
    std::vector<double> xb = s.b, pi(rows);
    std::vector<bool> basic(cols);
    for (auto j : s.basis)
        basic[j] = true;
    auto checkpoint = [&]() {
        r.x = s.shift;
        r.y.assign(m.rl.size(), 0);
        for (int i = 0; i < rows; ++i) {
            auto j = s.basis[i];
            if (s.original[j] >= 0)
                r.x[s.original[j]] += s.sign[j] * xb[i];
            if (s.row_map[i] >= 0)
                r.y[s.row_map[i]] -= s.row_sign[i] * pi[i];
        }
        auto tick = Clock::now();
        r.accuracy = verify(m, r.x, r.y);
        r.verification_seconds += elapsed(tick);
    };
    auto pivot = [&](int entering, int leaving, std::vector<double> direction) {
        basic[s.basis[leaving]] = false;
        s.basis[leaving] = entering;
        basic[entering] = true;
        etas.push_back({leaving, std::move(direction)});
        if (etas.size() >= 40) {
            factor.factor(s.columns, s.basis);
            etas.clear();
        }
    };
    try {
        for (int phase = 1; phase <= 2; ++phase) {
            std::vector<double> cost = s.c;
            if (phase == 1)
                for (int j = 0; j < cols; ++j)
                    cost[j] = s.artificial[j] ? 1 : 0;
            int degenerate_pivots = 0;
            while (true) {
                if (interrupted) {
                    r.status = "INTERRUPTED";
                    checkpoint();
                    return r;
                }
                if (elapsed(start) >= o.time_limit) {
                    r.status = "TIME_LIMIT";
                    checkpoint();
                    return r;
                }
                if (r.iterations >= o.iteration_limit) {
                    r.status = "ITERATION_LIMIT";
                    checkpoint();
                    return r;
                }
                xb = solveB(s.b);
                double activity_scale = 1;
                for (auto value : s.b)
                    activity_scale = std::max(activity_scale, std::abs(value));
                bool lost_feasibility = std::any_of(xb.begin(), xb.end(), [&](double value) {
                    return !std::isfinite(value) || value < -1e-8 * activity_scale;
                });
                if (lost_feasibility) {
                    factor.factor(s.columns, s.basis);
                    etas.clear();
                    xb = solveB(s.b);
                    if (std::any_of(xb.begin(), xb.end(), [&](double value) {
                            return !std::isfinite(value) || value < -1e-8 * activity_scale;
                        }))
                        throw std::runtime_error(
                            "Simplex basis lost primal feasibility after refactorization");
                }
                std::vector<double> cb(rows);
                for (int i = 0; i < rows; ++i)
                    cb[i] = cost[s.basis[i]];
                pi = solveB(cb, true);
                int entering = -1;
                double best = -1e-10;
                for (int j = 0; j < cols; ++j)
                    if (!basic[j] && (phase == 1 || !s.artificial[j])) {
                        double rc = cost[j];
                        for (auto k = s.columns.ptr[j]; k < s.columns.ptr[j + 1]; ++k)
                            rc -= s.columns.value[k] * pi[s.columns.index[k]];
                        if (rc < best) {
                            best = rc;
                            entering = j;
                            if (degenerate_pivots >= 20)
                                break; // Bland fallback on sustained degeneracy.
                        }
                    }
                if (entering < 0)
                    break;
                std::vector<double> rhs(rows);
                for (auto k = s.columns.ptr[entering]; k < s.columns.ptr[entering + 1]; ++k)
                    rhs[s.columns.index[k]] = s.columns.value[k];
                auto d = solveB(rhs);
                int leaving = -1;
                double ratio = inf;
                for (int i = 0; i < rows; ++i)
                    if (d[i] > 1e-12) {
                        double q = std::max(0., xb[i]) / d[i];
                        if (q < ratio - 1e-12 || (std::abs(q - ratio) <= 1e-12 &&
                                                  (leaving < 0 || s.basis[i] < s.basis[leaving]))) {
                            ratio = q;
                            leaving = i;
                        }
                    }
                if (leaving < 0) {
                    r.status = "UNKNOWN";
                    r.message = "Improving ray found; general recession certification required";
                    checkpoint();
                    return r;
                }
                if (ratio <= 1e-12)
                    ++degenerate_pivots;
                else
                    degenerate_pivots = 0;
                pivot(entering, leaving, std::move(d));
                r.iterations++;
            }
            if (phase == 1) {
                long double residual = 0;
                for (int i = 0; i < rows; ++i)
                    if (s.artificial[s.basis[i]])
                        residual += std::max(0., xb[i]);
                if (residual > 1e-8) {
                    checkpoint();
                    Verifier verifier(m);
                    auto ray = r.y;
                    double norm = 0;
                    for (auto v : ray)
                        norm = std::max(norm, std::abs(v));
                    if (norm > 0)
                        for (auto &v : ray)
                            v /= norm;
                    auto margin = verifier.infeasibility_bound(ray);
                    r.status = margin > 0 ? "INFEASIBLE" : "UNKNOWN";
                    if (margin > 0) {
                        r.infeasibility_ray = std::move(ray);
                        r.certificate_margin = margin;
                    }
                    return r;
                }
                // Degenerate pivots remove artificial basics. If every non-artificial
                // column has zero transformed coefficient, that row is redundant.
                for (int i = 0; i < rows; ++i)
                    if (s.artificial[s.basis[i]])
                        for (int j = 0; j < cols; ++j)
                            if (!basic[j] && !s.artificial[j]) {
                                if (interrupted || elapsed(start) >= o.time_limit) {
                                    r.status = interrupted ? "INTERRUPTED" : "TIME_LIMIT";
                                    checkpoint();
                                    return r;
                                }
                                std::vector<double> rhs(rows);
                                for (auto k = s.columns.ptr[j]; k < s.columns.ptr[j + 1]; ++k)
                                    rhs[s.columns.index[k]] = s.columns.value[k];
                                auto d = solveB(rhs);
                                if (std::abs(d[i]) > 1e-10) {
                                    pivot(j, i, std::move(d));
                                    break;
                                }
                            }
            }
        }
        xb = solveB(s.b);
        std::vector<double> cb(rows);
        for (int i = 0; i < rows; ++i)
            cb[i] = s.c[s.basis[i]];
        pi = solveB(cb, true);
        checkpoint();
        r.status = r.accuracy.finite && r.accuracy.kkt <= o.tol ? "OPTIMAL" : "NUMERICAL_ERROR";
    } catch (const std::exception &e) {
        r.status = "NUMERICAL_ERROR";
        r.message = e.what();
    }
    r.iteration_seconds = elapsed(start);
    r.seconds = r.iteration_seconds;
    return r;
}
Result solve_simplex(const Model &original, const Options &o) {
    if (original.is_qp() || o.device == "cuda")
        return solve_simplex_raw(original, o);
    auto start = Clock::now();
    auto p = prepare(original, o);
    auto preprocess_seconds = elapsed(start);
    if (!p.failure.empty()) {
        Result r;
        r.status = p.failure;
        r.message = p.reason;
        r.method_selected = "revised-primal-simplex";
        r.preprocess_seconds = r.seconds = preprocess_seconds;
        return r;
    }
    auto options = o;
    options.time_limit = std::max(0., o.time_limit - preprocess_seconds);
    auto r = solve_simplex_raw(p.model, options);
    r.preprocess_seconds += preprocess_seconds;
    r.removed_columns = original.c.size() - p.model.c.size();
    r.removed_rows = original.rl.size() - p.model.rl.size();
    auto verification_start = Clock::now();
    if (r.x.size() == p.model.c.size() && r.y.size() == p.model.rl.size()) {
        r.x = p.restore_x(r.x);
        r.y = p.restore_y(r.y, original.rl.size());
        r.accuracy = verify(original, r.x, r.y);
        if (r.status == "OPTIMAL" && (!r.accuracy.finite || r.accuracy.kkt > o.tol))
            r.status = "NUMERICAL_ERROR";
    }
    if (!r.infeasibility_ray.empty()) {
        r.infeasibility_ray = p.restore_y(r.infeasibility_ray, original.rl.size());
        r.certificate_margin = Verifier(original).infeasibility_bound(r.infeasibility_ray);
        if (!(r.certificate_margin > 0)) {
            r.status = "UNKNOWN";
            r.infeasibility_ray.clear();
        }
    }
    r.verification_seconds += elapsed(verification_start);
    r.seconds = elapsed(start);
    return r;
}
} // namespace vantage
