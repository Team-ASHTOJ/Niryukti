#include "checkpoint.hpp"
#include "internal.hpp"
#include <iostream>
#include <stdexcept>
#ifdef VANTAGE_SPARSE_LU
#include <Eigen/SparseLU>
#endif
namespace vantage {
Result solve_barrier(const Model &original, const Options &o) {
    Result result;
    result.method_selected = "predictor-corrector-barrier";
    result.device_reason = "CPU sparse Newton factorization";
#ifndef VANTAGE_SPARSE_LU
    result.status = "UNSUPPORTED";
    result.message = "Barrier requires sparse numerical factorization";
    return result;
#else
    bool gpu = gpu_request(o);
    if (gpu && !cuda_available()) {
        result.status = "UNSUPPORTED";
        result.message = "CUDA device unavailable";
        return result;
    }
    if (gpu) {
        result.backend = gpu_backend_name();
        result.device_reason = "GPU sparse BiCGSTAB Newton solves; CPU KKT assembly";
    }
    auto start = Clock::now();
    auto prepared = prepare(original, o);
    result.preprocess_seconds = elapsed(start);
    if (!prepared.failure.empty()) {
        result.status = prepared.failure;
        result.message = prepared.reason;
        result.infeasibility_ray = prepared.failure_ray;
        if (!result.infeasibility_ray.empty())
            result.certificate_margin =
                Verifier(original).infeasibility_bound(result.infeasibility_ray);
        return result;
    }
    auto &m = prepared.model;
    using Matrix = Eigen::SparseMatrix<double>;
    using Vector = Eigen::VectorXd;
    using Triplet = Eigen::Triplet<double>;
    int n = int(m.c.size());
    if (m.c.size() > 16384 || m.rl.size() > 16384) {
        result.status = "UNSUPPORTED";
        result.message = "Barrier prototype dimension guard (16384)";
        return result;
    }
    std::vector<Triplet> ge, ee, he;
    std::vector<double> rhs_g, rhs_e;
    std::vector<std::pair<int, double>> map_g, map_e;
    long double fill_estimate = 0;
    auto row = [&](int i, double sign, bool equality) {
        int r = int(equality ? rhs_e.size() : rhs_g.size());
        auto &entries = equality ? ee : ge;
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
            entries.emplace_back(r, int(m.A.index[k]), sign * m.A.value[k]);
        if (equality) {
            rhs_e.push_back(m.rl[i]);
            map_e.push_back({i, 1});
        } else {
            rhs_g.push_back(sign > 0 ? m.ru[i] : -m.rl[i]);
            map_g.push_back({i, sign});
            auto degree = m.A.ptr[i + 1] - m.A.ptr[i];
            fill_estimate += (long double)degree * degree;
        }
    };
    for (int i = 0; i < int(m.rl.size()); ++i) {
        if (m.rl[i] == m.ru[i])
            row(i, 1, true);
        else {
            if (std::isfinite(m.ru[i]))
                row(i, 1, false);
            if (std::isfinite(m.rl[i]))
                row(i, -1, false);
        }
    }
    for (int j = 0; j < n; ++j) {
        he.emplace_back(j, j, m.q[j]);
        if (!m.Q.value.empty())
            for (auto k = m.Q.ptr[j]; k < m.Q.ptr[j + 1]; ++k)
                he.emplace_back(j, int(m.Q.index[k]), m.Q.value[k]);
        if (m.lb[j] == m.ub[j]) {
            ee.emplace_back(int(rhs_e.size()), j, 1);
            rhs_e.push_back(m.lb[j]);
            map_e.push_back({-1, 0});
        } else {
            if (std::isfinite(m.ub[j])) {
                ge.emplace_back(int(rhs_g.size()), j, 1);
                rhs_g.push_back(m.ub[j]);
                map_g.push_back({-1, 0});
            }
            if (std::isfinite(m.lb[j])) {
                ge.emplace_back(int(rhs_g.size()), j, -1);
                rhs_g.push_back(-m.lb[j]);
                map_g.push_back({-1, 0});
            }
        }
    }
    if (fill_estimate > 10000000 || n + rhs_e.size() > 16384) {
        result.status = "UNSUPPORTED";
        result.message = "Barrier estimated Newton fill exceeds prototype guard";
        return result;
    }
    int inequalities = int(rhs_g.size()), equalities = int(rhs_e.size());
    Matrix g(inequalities, n), e(equalities, n), h(n, n);
    g.setFromTriplets(ge.begin(), ge.end());
    e.setFromTriplets(ee.begin(), ee.end());
    h.setFromTriplets(he.begin(), he.end());
    Vector b = Eigen::Map<Vector>(rhs_e.data(), equalities),
           bound = Eigen::Map<Vector>(rhs_g.data(), inequalities);
    Vector c = Eigen::Map<Vector>(m.c.data(), n), x(n), y = Vector::Zero(equalities),
           z = Vector::Ones(inequalities);
    for (int j = 0; j < n; ++j) {
        x[j] =
            o.initial_x.empty() ? 0 : o.initial_x.at(prepared.cols[j]) / prepared.column_scale[j];
        if (std::isfinite(m.lb[j]) && std::isfinite(m.ub[j]))
            x[j] = .5 * m.lb[j] + .5 * m.ub[j];
        else
            x[j] = std::clamp(x[j], m.lb[j], m.ub[j]);
    }
    Vector slack = (bound - g * x).cwiseMax(1.);
    if (!o.resume_path.empty()) {
        auto saved =
            read_engine_checkpoint(o.resume_path, "niryukti-barrier-1", original.fingerprint(), o);
        auto restore = [&](const char *name, Vector &target, bool positive) {
            auto values = saved.at(name).get<std::vector<double>>();
            if (values.size() != size_t(target.size()))
                throw std::runtime_error("Barrier checkpoint dimensions");
            for (auto v : values)
                if (!std::isfinite(v) || (positive && !(v > 0)))
                    throw std::runtime_error("Invalid barrier checkpoint vector");
            target = Eigen::Map<Vector>(values.data(), values.size());
        };
        restore("x", x, false);
        restore("y", y, false);
        restore("z", z, true);
        restore("slack", slack, true);
        result.iterations = saved.at("iterations");
        result.x = saved.at("best_x").get<std::vector<double>>();
        result.y = saved.at("best_y").get<std::vector<double>>();
        if (!result.x.empty() || !result.y.empty()) {
            if (result.x.size() != original.c.size() || result.y.size() != original.rl.size())
                throw std::runtime_error("Barrier checkpoint best-candidate dimensions");
            result.accuracy = verify(original, result.x, result.y);
            if (!result.accuracy.finite)
                throw std::runtime_error("Invalid barrier checkpoint candidate");
        }
    }
    auto save = [&]() {
        auto values = [](const Vector &v) {
            return std::vector<double>(v.data(), v.data() + v.size());
        };
        write_engine_checkpoint(o.checkpoint_path, {{"schema", "niryukti-barrier-1"},
                                                    {"fingerprint", original.fingerprint()},
                                                    {"configuration", checkpoint_configuration(o)},
                                                    {"iterations", result.iterations},
                                                    {"best_x", result.x},
                                                    {"best_y", result.y},
                                                    {"x", values(x)},
                                                    {"y", values(y)},
                                                    {"z", values(z)},
                                                    {"slack", values(slack)}});
    };
    auto candidate = [&]() {
        std::vector<double> dual(m.rl.size(), 0);
        for (int i = 0; i < equalities; ++i)
            if (map_e[i].first >= 0)
                dual[map_e[i].first] += y[i];
        for (int i = 0; i < inequalities; ++i)
            if (map_g[i].first >= 0)
                dual[map_g[i].first] += map_g[i].second * z[i];
        auto primal = prepared.restore_x(std::vector<double>(x.data(), x.data() + n));
        auto restored = prepared.restore_y(dual, original.rl.size());
        auto accuracy = verify(original, primal, restored);
        if (accuracy.finite && accuracy.kkt < result.accuracy.kkt) {
            result.x = std::move(primal);
            result.y = std::move(restored);
            result.accuracy = accuracy;
        }
        return accuracy.finite && accuracy.kkt <= o.tol;
    };
    auto step = [](const Vector &value, const Vector &direction, double fraction) {
        double alpha = 1;
        for (int i = 0; i < value.size(); ++i)
            if (direction[i] < 0)
                alpha = std::min(alpha, -fraction * value[i] / direction[i]);
        return alpha;
    };
    result.status = "ITERATION_LIMIT";
    try {
        for (int64_t iteration = result.iterations; iteration <= o.iteration_limit; ++iteration) {
            if (iteration % o.checkpoint_nodes == 0)
                save();
            if (candidate()) {
                result.status = "OPTIMAL";
                break;
            }
            if (stop_requested(o)) {
                result.status = "INTERRUPTED";
                break;
            }
            if (elapsed(start) >= o.time_limit) {
                result.status = "TIME_LIMIT";
                break;
            }
            if (iteration == o.iteration_limit)
                break;
            Vector rd = h * x + c + e.transpose() * y + g.transpose() * z;
            Vector rp = e * x - b, rg = g * x + slack - bound;
            Vector product = slack.cwiseProduct(z);
            double mu = inequalities ? product.mean() : 0;
            if (!x.allFinite() || !slack.allFinite() || !z.allFinite() ||
                (inequalities && (slack.minCoeff() <= 0 || z.minCoeff() <= 0)))
                throw std::runtime_error("Nonfinite/nonpositive barrier state");
            Matrix weighted = g;
            for (int col = 0; col < weighted.outerSize(); ++col)
                for (Matrix::InnerIterator k(weighted, col); k; ++k)
                    k.valueRef() *= z[k.row()] / slack[k.row()];
            Matrix normal = h + g.transpose() * weighted;
            std::vector<Triplet> kt;
            for (int col = 0; col < normal.outerSize(); ++col)
                for (Matrix::InnerIterator k(normal, col); k; ++k)
                    kt.emplace_back(k.row(), k.col(), k.value());
            double scale = 1;
            for (int j = 0; j < n; ++j)
                scale = std::max(scale, std::abs(normal.coeff(j, j)));
            double regularization = 1e-12 * scale;
            for (int j = 0; j < n; ++j)
                kt.emplace_back(j, j, regularization);
            for (int col = 0; col < e.outerSize(); ++col)
                for (Matrix::InnerIterator k(e, col); k; ++k) {
                    kt.emplace_back(n + k.row(), k.col(), k.value());
                    kt.emplace_back(k.col(), n + k.row(), k.value());
                }
            for (int i = 0; i < equalities; ++i)
                kt.emplace_back(n + i, n + i, -1e-12);
            Matrix kkt(n + equalities, n + equalities);
            kkt.setFromTriplets(kt.begin(), kt.end());
            Eigen::SparseLU<Matrix> factor;
            if (!gpu)
                factor.compute(kkt);
            if (!gpu && factor.info() != Eigen::Success)
                throw std::runtime_error("Barrier Newton factorization failed");
            auto direction = [&](const Vector &rc, Vector &dx, Vector &dy, Vector &ds, Vector &dz) {
                Vector rhs(n + equalities);
                rhs.head(n) =
                    -rd + g.transpose() * ((rc - z.cwiseProduct(rg)).cwiseQuotient(slack));
                rhs.tail(equalities) = -rp;
                Vector solution;
                if (gpu) {
                    std::vector<Entry> entries;
                    for (int col = 0; col < kkt.outerSize(); ++col)
                        for (Matrix::InnerIterator k(kkt, col); k; ++k)
                            entries.push_back({k.row(), k.col(), k.value()});
                    auto sparse = Sparse::build(kkt.rows(), kkt.cols(), std::move(entries));
                    auto budget = o;
                    budget.time_limit = std::max(0., o.time_limit - elapsed(start));
                    auto answer = cuda_linear_solve(
                        sparse, std::vector<double>(rhs.data(), rhs.data() + rhs.size()), budget);
                    solution = Eigen::Map<Vector>(answer.data(), answer.size());
                } else
                    solution = factor.solve(rhs);
                if (!solution.allFinite())
                    throw std::runtime_error("Nonfinite barrier Newton direction");
                double residual = (kkt * solution - rhs).lpNorm<Eigen::Infinity>();
                if (residual > 1e-7 * (1 + rhs.lpNorm<Eigen::Infinity>())) {
                    if (gpu)
                        throw std::runtime_error(
                            "GPU Newton direction residual exceeds barrier threshold");
                    Vector correction = factor.solve(rhs - kkt * solution);
                    solution += correction;
                    if (!solution.allFinite() || (kkt * solution - rhs).lpNorm<Eigen::Infinity>() >
                                                     1e-6 * (1 + rhs.lpNorm<Eigen::Infinity>()))
                        throw std::runtime_error("Barrier Newton solve residual too large");
                }
                dx = solution.head(n);
                dy = solution.tail(equalities);
                ds = -rg - g * dx;
                dz = (-rc - z.cwiseProduct(ds)).cwiseQuotient(slack);
            };
            Vector dx, dy, ds, dz;
            direction(product, dx, dy, ds, dz);
            if (inequalities) {
                double ap = step(slack, ds, 1), ad = step(z, dz, 1);
                double affine = (slack + ap * ds).dot(z + ad * dz) / inequalities;
                double sigma =
                    std::clamp(std::pow(std::max(0., affine) / std::max(mu, 1e-300), 3.), 0., 1.);
                Vector rc =
                    product + ds.cwiseProduct(dz) - Vector::Constant(inequalities, sigma * mu);
                direction(rc, dx, dy, ds, dz);
            }
            double ap = step(slack, ds, .995), ad = step(z, dz, .995);
            x += ap * dx;
            slack += ap * ds;
            y += ad * dy;
            z += ad * dz;
            result.iterations++;
            if (o.verbose)
                std::cerr << "barrier_iter=" << result.iterations << " mu=" << mu << '\n';
        }
    } catch (const std::exception &error) {
        result.status = "NUMERICAL_ERROR";
        result.message = error.what();
    }
    save();
    result.iteration_seconds = std::max(0., elapsed(start) - result.preprocess_seconds);
    result.seconds = elapsed(start);
    return result;
#endif
}
} // namespace vantage
