#include "internal.hpp"
#include <numeric>
namespace vantage {
namespace {
long double outward(long double x, bool lower) {
    return std::nextafter(x, lower ? -inf : inf);
}
} // namespace
BoundPostsolve prove_gpu_bounds(const Model &m, const std::vector<double> &target_lower,
                                const std::vector<double> &target_upper, const Options &o) {
    BoundPostsolve proof;
    proof.lower = m.lb;
    proof.upper = m.ub;
    const auto n = m.c.size();
    proof.lower_sources.resize(n);
    proof.upper_sources.resize(n);
    for (size_t j = 0; j < n; ++j) {
        proof.lower_sources[j] = 2 * j;
        proof.upper_sources[j] = 2 * j + 1;
    }
    if (target_lower.size() != n || target_upper.size() != n)
        return proof;
    auto start = Clock::now();
    size_t dependencies = 0;
    for (int pass = 0; pass < 5 && !stop_requested(o) && elapsed(start) < o.time_limit; ++pass) {
        auto lo = proof.lower, hi = proof.upper;
        auto low_sources = proof.lower_sources, high_sources = proof.upper_sources;
        bool changed = false;
        for (int64_t row = 0; row < m.A.rows; ++row) {
            // A proof records dependencies explicitly; cap dense rows rather
            // than turning sparse GPU presolve into unbounded quadratic host work.
            if (m.A.ptr[row + 1] - m.A.ptr[row] > 4096)
                continue;
            if (stop_requested(o) || elapsed(start) >= o.time_limit)
                break;
            for (auto k = m.A.ptr[row]; k < m.A.ptr[row + 1]; ++k) {
                auto j = m.A.index[k];
                auto a = m.A.value[k];
                if (a == 0 || !std::isfinite(1 / a))
                    continue;
                for (bool row_upper : {false, true}) {
                    double rhs = row_upper ? m.ru[row] : m.rl[row];
                    if (!std::isfinite(rhs))
                        continue;
                    bool lower = row_upper ? a < 0 : a > 0;
                    if (lower ? !(target_lower[j] > proof.lower[j])
                              : !(target_upper[j] < proof.upper[j]))
                        continue;
                    if (stop_requested(o) || elapsed(start) >= o.time_limit)
                        break;
                    long double activity = 0;
                    bool finite = true;
                    BoundDerivation derivation{j, row, lower, (lower ? -1. : 1.) / a, {}};
                    for (auto t = m.A.ptr[row]; t < m.A.ptr[row + 1]; ++t) {
                        auto other = m.A.index[t];
                        double coefficient = m.A.value[t];
                        if (other == j || coefficient == 0)
                            continue;
                        bool use_lower = row_upper ? coefficient > 0 : coefficient < 0;
                        double endpoint = use_lower ? lo[other] : hi[other];
                        if (!std::isfinite(endpoint)) {
                            finite = false;
                            break;
                        }
                        auto term = outward((long double)coefficient * endpoint, row_upper);
                        activity = outward(activity + term, row_upper);
                        double weight = std::abs(derivation.row_multiplier * coefficient);
                        if (!std::isfinite(weight) || weight == 0) {
                            finite = false;
                            break;
                        }
                        derivation.dependencies.push_back(
                            {use_lower ? low_sources[other] : high_sources[other], weight});
                    }
                    if (!finite)
                        continue;
                    // For a lower row, overestimate all other terms and subtract
                    // downward. For an upper row, underestimate and subtract upward.
                    auto numerator = outward((long double)rhs - activity, !row_upper);
                    auto bound = outward(numerator / a, lower);
                    double value = std::nextafter(double(bound), lower ? -inf : inf);
                    value =
                        lower ? std::min(value, target_lower[j]) : std::max(value, target_upper[j]);
                    if (!std::isfinite(value) || value < m.lb[j] || value > m.ub[j] ||
                        (lower ? value <= proof.lower[j] || value > proof.upper[j]
                               : value >= proof.upper[j] || value < proof.lower[j]))
                        continue;
                    if (proof.derivations.size() >= 200000 ||
                        dependencies + derivation.dependencies.size() > 2000000)
                        continue;
                    auto id = int64_t(2 * n + proof.derivations.size());
                    dependencies += derivation.dependencies.size();
                    proof.derivations.push_back(std::move(derivation));
                    (lower ? proof.lower : proof.upper)[j] = value;
                    (lower ? proof.lower_sources : proof.upper_sources)[j] = id;
                    changed = true;
                }
            }
        }
        if (!changed)
            break;
    }
    if (!proof.derivations.empty())
        proof.transpose = m.A.transpose();
    return proof;
}
std::vector<double> BoundPostsolve::lift_dual(const Model &m, const std::vector<double> &x,
                                              const std::vector<double> &y, bool objective) const {
    if (derivations.empty())
        return y;
    if (x.size() != m.c.size() || y.size() != m.rl.size())
        throw std::runtime_error("Bound postsolve dimensions");
    auto gradient = transpose.multiply(y);
    if (objective) {
        auto qx = m.Q.value.empty() ? std::vector<double>(x.size(), 0) : m.Q.multiply(x);
        for (size_t j = 0; j < x.size(); ++j)
            gradient[j] += m.c[j] + m.q[j] * x[j] + qx[j];
    }
    std::vector<long double> multipliers(2 * x.size() + derivations.size(), 0);
    for (size_t j = 0; j < x.size(); ++j)
        multipliers[gradient[j] >= 0 ? lower_sources[j] : upper_sources[j]] +=
            std::abs(gradient[j]);
    std::vector<long double> rows(y.begin(), y.end());
    for (size_t index = derivations.size(); index-- > 0;) {
        auto amount = multipliers[2 * x.size() + index];
        const auto &d = derivations[index];
        rows[d.row] += amount * d.row_multiplier;
        for (auto [dependency, weight] : d.dependencies)
            multipliers[dependency] += amount * weight;
    }
    return std::vector<double>(rows.begin(), rows.end());
}
} // namespace vantage
