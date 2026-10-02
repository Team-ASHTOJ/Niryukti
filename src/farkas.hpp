#pragma once
// Independent Farkas certification helpers shared by IIS search and global optimization.
#include "vantage/vantage.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
namespace vantage {
// Exact sign of sum_k a_k b_k via error-free transformations (TwoProduct with fma,
// Shewchuk's Grow-Expansion). Components grow in magnitude; the last nonzero gives the sign.
inline int exact_sign(const std::vector<std::pair<double, double>> &terms) {
    std::vector<double> e;
    auto grow = [&](double b) {
        double q = b;
        std::vector<double> h;
        for (double v : e) {
            double sum = q + v, bv = sum - q, err = (q - (sum - bv)) + (v - bv);
            if (err != 0)
                h.push_back(err);
            q = sum;
        }
        if (q != 0)
            h.push_back(q);
        e.swap(h);
    };
    for (auto [a, b] : terms) {
        double p = a * b;
        grow(p);
        grow(std::fma(a, b, -p));
    }
    return e.empty() ? 0 : (e.back() > 0 ? 1 : -1);
}
// Feasibility Farkas margin  -sum_i y_i b_i + sum_j min_{x_j in box} (A'y)_j x_j  with
// outward rounding. Unlike interval-only evaluation, a column whose reduced cost cancels
// exactly contributes zero even when its box is unbounded.
inline double farkas_margin(const Model &m, const Sparse &at, const std::vector<double> &y) {
    auto down = [](double v) { return std::nextafter(v, -inf); };
    auto up = [](double v) { return std::nextafter(v, inf); };
    double lower = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i]))
            return -inf;
        if (y[i] == 0)
            continue;
        double b = y[i] > 0 ? m.ru[i] : m.rl[i];
        if (!std::isfinite(b))
            return -inf;
        if (b != 0)
            lower = down(lower - up(y[i] * b));
    }
    std::vector<std::pair<double, double>> terms;
    for (int64_t j = 0; j < at.rows; ++j) {
        terms.clear();
        double gl = 0, gu = 0;
        for (auto k = at.ptr[j]; k < at.ptr[j + 1]; ++k)
            if (y[at.index[k]] != 0 && at.value[k] != 0) {
                double v = at.value[k] * y[at.index[k]];
                gl = down(gl + down(v));
                gu = up(gu + up(v));
                terms.push_back({at.value[k], y[at.index[k]]});
            }
        int sign = exact_sign(terms);
        if (sign == 0)
            continue;
        if ((sign > 0 && !std::isfinite(m.lb[j])) || (sign < 0 && !std::isfinite(m.ub[j])))
            return -inf;
        if (sign > 0)
            gl = std::max(gl, 0.);
        else
            gu = std::min(gu, 0.);
        double term = inf;
        for (double b : {m.lb[j], m.ub[j]})
            if (std::isfinite(b))
                term = std::min({term, b == 0 ? 0. : down(gl * b), b == 0 ? 0. : down(gu * b)});
        lower = down(lower + term);
    }
    return std::isfinite(lower) ? lower : -inf;
}
// Snap noisy rays to dyadic grids; a snapped ray is kept only if its margin verifies.
inline double polish(const Model &m, const Sparse &at, std::vector<double> &ray) {
    double best = farkas_margin(m, at, ray);
    if (best > 0)
        return best;
    double largest = 0, smallest = inf;
    for (double v : ray)
        if (std::abs(v) > 1e-9) {
            largest = std::max(largest, std::abs(v));
            smallest = std::min(smallest, std::abs(v));
        }
    if (largest == 0)
        return best;
    for (double unit : {smallest, largest})
        for (int k = 0; k <= 30; ++k) {
            double grid = std::ldexp(1., k);
            std::vector<double> candidate(ray.size());
            for (size_t i = 0; i < ray.size(); ++i)
                candidate[i] = std::round(ray[i] / unit * grid) / grid;
            double margin = farkas_margin(m, at, candidate);
            if (margin > 0) {
                ray = candidate;
                return margin;
            }
        }
    return best;
}
// Certifies infeasibility of `m` from a solver result: the returned ray (polished), else
// single-row rays, which cover presolve's row-activity proofs. Returns the verified margin
// (> 0) and the ray, or -inf.
inline double certify_infeasible(const Model &m, const Result &r, std::vector<double> *out = nullptr) {
    auto at = m.A.transpose();
    if (r.infeasibility_ray.size() == m.rl.size()) {
        auto ray = r.infeasibility_ray;
        double margin = polish(m, at, ray);
        if (margin > 0) {
            if (out)
                *out = ray;
            return margin;
        }
    }
    std::vector<double> ray(m.rl.size(), 0.);
    for (size_t i = 0; i < m.rl.size(); ++i)
        for (double sign : {1., -1.}) {
            ray[i] = sign;
            double margin = farkas_margin(m, at, ray);
            if (margin > 0) {
                if (out)
                    *out = ray;
                return margin;
            }
            ray[i] = 0;
        }
    return -inf;
}
} // namespace vantage
