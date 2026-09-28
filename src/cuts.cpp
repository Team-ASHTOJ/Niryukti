#include "internal.hpp"
#include <numeric>
#include <set>
#include <stdexcept>
namespace vantage {
bool propagate_bound_conflicts(const std::vector<std::vector<BoundLiteral>> &clauses,
                               const std::vector<VarType> &types, std::vector<double> &lb,
                               std::vector<double> &ub, int64_t &tightened) {
    for (size_t pass = 0; pass <= clauses.size(); ++pass) {
        bool changed = false;
        for (const auto &clause : clauses) {
            size_t unknown = 0;
            BoundLiteral unit{};
            bool satisfied = false;
            for (const auto &literal : clause) {
                auto j = literal.variable;
                if (literal.lower ? ub[j] < literal.value : lb[j] > literal.value) {
                    satisfied = true;
                    break;
                }
                if (!(literal.lower ? lb[j] >= literal.value : ub[j] <= literal.value)) {
                    ++unknown;
                    unit = literal;
                }
            }
            if (satisfied)
                continue;
            if (!unknown)
                return false;
            if (unknown == 1) {
                auto j = unit.variable;
                bool exact_integer = types[j] != VarType::Continuous &&
                                     std::abs(unit.value) < 0x1p52 &&
                                     unit.value == std::floor(unit.value);
                // Continuous negation is strict; its weak closed relaxation is
                // safe for tightening. We never substitute a guessed epsilon.
                double value = unit.value + (exact_integer ? (unit.lower ? -1. : 1.) : 0.);
                double &bound = unit.lower ? ub[j] : lb[j];
                double next = unit.lower ? std::min(bound, value) : std::max(bound, value);
                if (next != bound) {
                    bound = next;
                    changed = true;
                    ++tightened;
                }
                if (lb[j] > ub[j])
                    return false;
            }
        }
        if (!changed)
            break;
    }
    return true;
}
bool propagate_binary_conflicts(const std::vector<std::vector<std::pair<int64_t, int>>> &clauses,
                                std::vector<double> &lb, std::vector<double> &ub,
                                int64_t &tightened) {
    for (size_t pass = 0; pass <= clauses.size(); ++pass) {
        bool changed = false;
        for (const auto &clause : clauses) {
            int64_t undecided = -1;
            int forbidden = 0, count = 0;
            bool satisfied = false;
            for (auto [j, v] : clause) {
                if (v < lb[j] || v > ub[j]) {
                    satisfied = true;
                    break;
                }
                if (lb[j] != v || ub[j] != v) {
                    undecided = j;
                    forbidden = v;
                    ++count;
                }
            }
            if (satisfied)
                continue;
            if (!count)
                return false;
            if (count == 1) {
                if (forbidden == 0)
                    lb[undecided] = 1;
                else
                    ub[undecided] = 0;
                ++tightened;
                changed = true;
            }
        }
        if (!changed)
            break;
    }
    return true;
}
int add_integer_lattice_cuts(Model &m, int limit, const std::vector<double> *point) {
    if (point && point->size() != m.c.size())
        throw std::runtime_error("Lattice cut point dimensions");
    if (limit <= 0)
        return 0;
    std::vector<Entry> entries;
    std::set<std::vector<double>> seen;
    const auto rows = m.A.rows;
    for (int64_t row = 0; row < rows; ++row) {
        std::vector<double> key;
        for (auto k = m.A.ptr[row]; k < m.A.ptr[row + 1]; ++k) {
            entries.push_back({row, m.A.index[k], m.A.value[k]});
            key.push_back(double(m.A.index[k]));
            key.push_back(m.A.value[k]);
        }
        key.push_back(m.rl[row]);
        key.push_back(m.ru[row]);
        seen.insert(key);
    }
    int added = 0;
    for (int64_t row = 0; row < rows && added < limit; ++row) {
        int64_t divisor = 0;
        bool valid = true;
        for (auto k = m.A.ptr[row]; k < m.A.ptr[row + 1]; ++k) {
            double a = m.A.value[k];
            if (m.types[m.A.index[k]] == VarType::Continuous || std::abs(a) > 0x1p30 ||
                a != std::floor(a)) {
                valid = false;
                break;
            }
            divisor = std::gcd(divisor, int64_t(std::abs(a)));
        }
        if (!valid || divisor <= 1)
            continue;
        double lower = std::isfinite(m.rl[row])
                           ? double(std::ceil(std::nextafter((long double)m.rl[row] / divisor,
                                                             -(long double)inf)))
                           : -inf;
        double upper = std::isfinite(m.ru[row])
                           ? double(std::floor(std::nextafter((long double)m.ru[row] / divisor,
                                                              (long double)inf)))
                           : inf;
        // Integer dot products occupy a lattice. Restrict to exactly representable
        // endpoints; no coefficient rounding or epsilon is used in the proof.
        if ((std::isfinite(lower) && std::abs(lower) >= 0x1p52) ||
            (std::isfinite(upper) && std::abs(upper) >= 0x1p52))
            continue;
        if (lower <= m.rl[row] / divisor && upper >= m.ru[row] / divisor)
            continue;
        // Keep each appended row well formed even when the integer lattice
        // proves an equality infeasible. The original lower row plus this
        // rounded upper row still supplies the contradiction to propagation.
        if (lower > upper)
            lower = -inf;
        std::vector<double> key;
        long double activity = 0;
        for (auto k = m.A.ptr[row]; k < m.A.ptr[row + 1]; ++k) {
            key.push_back(double(m.A.index[k]));
            key.push_back(m.A.value[k] / divisor);
            if (point)
                activity += (long double)(m.A.value[k] / divisor) * (*point)[m.A.index[k]];
        }
        key.push_back(lower);
        key.push_back(upper);
        if (seen.count(key) || (point && activity >= lower - 1e-7 && activity <= upper + 1e-7))
            continue;
        seen.insert(key);
        auto destination = int64_t(m.rl.size());
        for (auto k = m.A.ptr[row]; k < m.A.ptr[row + 1]; ++k)
            entries.push_back({destination, m.A.index[k], m.A.value[k] / divisor});
        m.rl.push_back(lower);
        m.ru.push_back(upper);
        m.row_names.push_back("niryukti_lattice_" + std::to_string(destination));
        ++added;
    }
    if (added)
        m.A = Sparse::build(m.rl.size(), m.c.size(), std::move(entries));
    return added;
}
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
