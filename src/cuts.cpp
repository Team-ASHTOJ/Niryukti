#include "internal.hpp"
#include <set>
#include <stdexcept>
namespace vantage {
int add_mir_cuts(Model &m, int limit, const std::vector<double> *point) {
    if (point && point->size() != m.c.size())
        throw std::runtime_error("MIR separation point dimension mismatch");
    if (limit <= 0)
        return 0;
    auto original_rows = m.A.rows;
    std::vector<Entry> all;
    for (int64_t i = 0; i < m.A.rows; ++i)
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
            all.push_back({i, m.A.index[k], m.A.value[k]});
    std::set<std::vector<double>> fingerprints;
    // Seed exact row fingerprints: repeated separation must not append the same cut.
    for (int64_t i = 0; i < original_rows; ++i) {
        if (!std::isfinite(m.rl[i]))
            continue;
        std::vector<double> key;
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k) {
            key.push_back(double(m.A.index[k]));
            key.push_back(m.A.value[k]);
        }
        key.push_back(m.rl[i]);
        fingerprints.insert(std::move(key));
    }
    int added = 0;
    for (int64_t row = 0; row < original_rows && added < limit; ++row)
        for (int side : {1, -1}) {
            double endpoint = side > 0 ? m.rl[row] : m.ru[row];
            if (!std::isfinite(endpoint))
                continue;
            long double rhs = side * endpoint;
            bool valid = true, has_integer = false;
            for (auto k = m.A.ptr[row]; k < m.A.ptr[row + 1]; ++k) {
                auto j = m.A.index[k];
                if (!std::isfinite(m.lb[j]) || std::abs(m.lb[j]) >= 0x1p30) {
                    valid = false;
                    break;
                }
                if (m.types[j] != VarType::Continuous) {
                    has_integer = true;
                    if (m.lb[j] != std::floor(m.lb[j])) {
                        valid = false;
                        break;
                    }
                }
                auto term = (long double)(side * m.A.value[k]) * m.lb[j];
                rhs = std::nextafter(rhs - std::nextafter(term, inf), -inf);
            }
            if (!valid || !has_integer || !std::isfinite(rhs) || std::abs(rhs) >= 0x1p30)
                continue;
            long double fractional = rhs - std::floor(rhs);
            if (fractional < 1e-6 || fractional > 1 - 1e-6)
                continue;
            std::vector<std::pair<int64_t, double>> cut;
            long double bound = std::ceil(rhs);
            std::vector<double> key;
            for (auto k = m.A.ptr[row]; k < m.A.ptr[row + 1]; ++k) {
                auto j = m.A.index[k];
                long double a = side * m.A.value[k], coefficient = 0;
                if (std::abs(a) >= 0x1p30) {
                    valid = false;
                    break;
                }
                if (m.types[j] == VarType::Continuous)
                    coefficient = std::max(0.L, a) / fractional;
                else
                    coefficient = std::floor(a) + std::min(1.L, (a - std::floor(a)) / fractional);
                // Shifted variables are nonnegative. Rounding coefficients upward and
                // the RHS downward produces a weaker, conservative inequality.
                double alpha = coefficient == 0 ? 0 : std::nextafter(double(coefficient), inf);
                if (!std::isfinite(alpha) || std::abs(alpha) > 1e12) {
                    valid = false;
                    break;
                }
                if (alpha != 0) {
                    cut.push_back({j, alpha});
                    key.push_back(double(j));
                    key.push_back(alpha);
                    bound = std::nextafter(
                        bound + std::nextafter((long double)alpha * m.lb[j], -inf), -inf);
                }
            }
            if (!valid || cut.empty() || !std::isfinite(bound))
                continue;
            double lower = std::nextafter(double(bound), -inf);
            if (point) {
                long double activity = 0, magnitude = std::abs((long double)lower);
                bool finite = true;
                for (auto [j, alpha] : cut) {
                    finite &= std::isfinite((*point)[j]);
                    long double term = (long double)alpha * (*point)[j];
                    activity += term;
                    magnitude += std::abs(term);
                }
                if (!finite || !std::isfinite(activity) ||
                    (long double)lower - activity <= 1e-7L * (1 + magnitude))
                    continue;
            }
            key.push_back(lower);
            if (!fingerprints.insert(key).second)
                continue;
            auto i = m.rl.size();
            for (auto [j, v] : cut)
                all.push_back({int64_t(i), j, v});
            m.rl.push_back(lower);
            m.ru.push_back(inf);
            auto name = "vantage_mir_" + std::to_string(i);
            while (std::find(m.row_names.begin(), m.row_names.end(), name) != m.row_names.end())
                name += "_";
            m.row_names.push_back(std::move(name));
            added++;
        }
    if (added)
        m.A = Sparse::build(m.rl.size(), m.c.size(), std::move(all));
    return added;
}
} // namespace vantage
