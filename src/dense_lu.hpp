#pragma once
// Dense LU with partial pivoting and mixed-precision iterative refinement.
// The basis is factorized in FP32; FP64 residuals r = b - Mx drive correction solves
// x += LU32^{-1} r (Wilkinson/Moler iterative refinement; Carson & Higham, SISC 2018).
// If FP32 factorization is singular or refinement stalls, the FP64 factor is used.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace vantage {
template <class T> class DenseFactor {
    int64_t n = 0;
    std::vector<T> lu; // column-major
    std::vector<int64_t> pivot;

  public:
    bool factor(const std::vector<double> &matrix, int64_t size) {
        n = size;
        lu.assign(matrix.begin(), matrix.end());
        pivot.resize(n);
        for (int64_t k = 0; k < n; ++k) {
            int64_t p = k;
            T best = std::abs(lu[k * n + k]);
            for (int64_t i = k + 1; i < n; ++i)
                if (std::abs(lu[k * n + i]) > best) {
                    best = std::abs(lu[k * n + i]);
                    p = i;
                }
            if (!(best > T(0)) || !std::isfinite(double(best)))
                return false;
            pivot[k] = p;
            if (p != k)
                for (int64_t j = 0; j < n; ++j)
                    std::swap(lu[j * n + k], lu[j * n + p]);
            T inverse = T(1) / lu[k * n + k];
            for (int64_t i = k + 1; i < n; ++i)
                lu[k * n + i] *= inverse;
            for (int64_t j = k + 1; j < n; ++j) {
                T f = lu[j * n + k];
                if (f == T(0))
                    continue;
                for (int64_t i = k + 1; i < n; ++i)
                    lu[j * n + i] -= lu[k * n + i] * f;
            }
        }
        return true;
    }
    // Solves M x = b (transpose=false) or M^T x = b (transpose=true) in place.
    void solve(std::vector<double> &b, bool transpose) const {
        std::vector<T> x(b.begin(), b.end());
        if (!transpose) {
            for (int64_t k = 0; k < n; ++k)
                if (pivot[k] != k)
                    std::swap(x[k], x[pivot[k]]);
            for (int64_t k = 0; k < n; ++k)
                for (int64_t i = k + 1; i < n; ++i)
                    x[i] -= lu[k * n + i] * x[k];
            for (int64_t k = n - 1; k >= 0; --k) {
                x[k] /= lu[k * n + k];
                for (int64_t i = 0; i < k; ++i)
                    x[i] -= lu[k * n + i] * x[k];
            }
        } else {
            for (int64_t k = 0; k < n; ++k) {
                for (int64_t i = 0; i < k; ++i)
                    x[k] -= lu[k * n + i] * x[i];
                x[k] /= lu[k * n + k];
            }
            for (int64_t k = n - 1; k >= 0; --k)
                for (int64_t i = k + 1; i < n; ++i)
                    x[k] -= lu[k * n + i] * x[i];
            for (int64_t k = n - 1; k >= 0; --k)
                if (pivot[k] != k)
                    std::swap(x[k], x[pivot[k]]);
        }
        for (int64_t i = 0; i < n; ++i)
            b[i] = double(x[i]);
    }
};

struct RefinementStats {
    int64_t solves = 0, refinement_steps = 0, fp64_fallbacks = 0;
    double worst_relative_residual = 0;
    bool fp32_factor = false, fp64_factor = false;
};

class MixedPrecisionSolver {
    int64_t n = 0;
    std::vector<double> matrix; // column-major FP64 original
    DenseFactor<float> low;
    DenseFactor<double> high;
    bool low_ok = false, high_ok = false;

    void residual(const std::vector<double> &x, const std::vector<double> &b, bool transpose,
                  std::vector<double> &r, double &scale) const {
        r = b;
        scale = 0;
        for (double v : b)
            scale = std::max(scale, std::abs(v));
        double xnorm = 0;
        for (double v : x)
            xnorm = std::max(xnorm, std::abs(v));
        double anorm = 0;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < n; ++i) {
                double a = matrix[j * n + i];
                anorm = std::max(anorm, std::abs(a));
                if (transpose)
                    r[j] -= a * x[i];
                else
                    r[i] -= a * x[j];
            }
        scale += anorm * xnorm * double(n);
    }
    bool ensure_high() {
        if (!high_ok) {
            high_ok = high.factor(matrix, n);
            stats.fp64_factor = high_ok;
        }
        return high_ok;
    }

  public:
    RefinementStats stats;
    bool factor(std::vector<double> column_major, int64_t size, bool mixed = true) {
        n = size;
        matrix = std::move(column_major);
        high_ok = false;
        low_ok = mixed && low.factor(matrix, n);
        stats.fp32_factor = low_ok;
        if (!low_ok)
            return ensure_high();
        return true;
    }
    std::vector<double> solve(const std::vector<double> &b, bool transpose = false) {
        ++stats.solves;
        std::vector<double> x = b, r;
        double scale = 0;
        if (low_ok) {
            low.solve(x, transpose);
            for (int step = 0; step < 30; ++step) {
                residual(x, b, transpose, r, scale);
                double worst = 0;
                for (double v : r)
                    worst = std::max(worst, std::abs(v));
                if (!std::isfinite(worst))
                    break;
                if (worst <= 1e-15 * std::max(scale, 1e-300)) {
                    stats.worst_relative_residual =
                        std::max(stats.worst_relative_residual, scale > 0 ? worst / scale : 0.);
                    return x;
                }
                low.solve(r, transpose);
                for (int64_t i = 0; i < n; ++i)
                    x[i] += r[i];
                ++stats.refinement_steps;
            }
            ++stats.fp64_fallbacks;
        }
        if (!ensure_high())
            throw std::runtime_error("Singular basis matrix");
        x = b;
        high.solve(x, transpose);
        for (int step = 0; step < 2; ++step) {
            residual(x, b, transpose, r, scale);
            high.solve(r, transpose);
            for (int64_t i = 0; i < n; ++i)
                x[i] += r[i];
        }
        residual(x, b, transpose, r, scale);
        double worst = 0;
        for (double v : r)
            worst = std::max(worst, std::abs(v));
        stats.worst_relative_residual =
            std::max(stats.worst_relative_residual, scale > 0 ? worst / scale : 0.);
        return x;
    }
};
} // namespace vantage
