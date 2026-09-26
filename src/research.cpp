#include "internal.hpp"
#include <random>
#include <set>
#include <stdexcept>
namespace vantage {
double power_norm(const Sparse &matrix, int iterations) {
    if (iterations < 1 || matrix.value.empty())
        return 0;
    double scale = 0;
    for (double v : matrix.value)
        scale = std::max(scale, std::abs(v));
    if (!(scale > 0) || !std::isfinite(scale))
        return 0;
    // Normalize the matrix first so A^T A cannot overflow simply from its units.
    Sparse a = matrix;
    for (double &v : a.value)
        v /= scale;
    auto at = a.transpose();
    std::mt19937_64 random(0x56414e54414745ULL);
    std::vector<double> x(a.cols);
    for (double &v : x)
        v = double(random() >> 11) * 0x1p-53 - .5;
    auto normalize = [](std::vector<double> &v) {
        double norm = 0;
        for (double value : v)
            norm = std::hypot(norm, value);
        if (!(norm > 0) || !std::isfinite(norm))
            return false;
        for (double &value : v)
            value /= norm;
        return true;
    };
    if (!normalize(x))
        return 0;
    double estimate = 0;
    for (int k = 0; k < iterations; ++k) {
        auto y = a.multiply(x);
        double norm = 0;
        for (double v : y)
            norm = std::hypot(norm, v);
        estimate = std::max(estimate, norm);
        if (!normalize(y))
            break;
        x = at.multiply(y);
        if (!normalize(x))
            break;
    }
    double result = scale * estimate;
    return std::isfinite(result) ? result : 0;
}

double PrimalWeightController::update(double weight, long double dx, long double dy) {
    if (!(weight > 0) || !std::isfinite(weight) || !(dx > 1e-24L) || !(dy > 1e-24L) ||
        !std::isfinite(dx) || !std::isfinite(dy))
        return weight;
    double error = double(std::log((long double)weight) + .5L * (std::log(dx) - std::log(dy)));
    error = std::clamp(error, -10., 10.);
    double next_integral = std::clamp(integral + error, -100., 100.);
    double derivative = initialized ? error - previous : 0;
    double correction =
        std::clamp(.3 * error + .01 * next_integral + .1 * derivative, -std::log(4.), std::log(4.));
    double logarithm = std::log(weight) - correction;
    double clipped = std::clamp(logarithm, std::log(1e-4), std::log(1e4));
    // Anti-windup: do not accumulate an error that pushes farther into saturation.
    if (clipped == logarithm)
        integral = next_integral;
    previous = error;
    initialized = true;
    return std::exp(clipped);
}

Model dual_feasibility_model(const Model &m) {
    if (m.is_qp())
        throw std::runtime_error("Dual feasibility polishing requires an LP");
    Model dual;
    dual.name = m.name + "_dual_feasibility";
    dual.A = m.A.transpose();
    dual.c.assign(m.rl.size(), 0);
    dual.q.assign(m.rl.size(), 0);
    dual.types.assign(m.rl.size(), VarType::Continuous);
    for (size_t i = 0; i < m.rl.size(); ++i) {
        dual.names.push_back("y" + std::to_string(i));
        // VANTAGE uses c + A^T y; upper-only multipliers are nonnegative.
        dual.lb.push_back(std::isfinite(m.rl[i]) ? -inf : 0);
        dual.ub.push_back(std::isfinite(m.ru[i]) ? inf : 0);
    }
    for (size_t j = 0; j < m.c.size(); ++j) {
        dual.row_names.push_back("stationarity" + std::to_string(j));
        dual.rl.push_back(std::isfinite(m.ub[j]) ? -inf : -m.c[j]);
        dual.ru.push_back(std::isfinite(m.lb[j]) ? inf : -m.c[j]);
    }
    dual.validate();
    return dual;
}

int add_binary_cuts(Model &m, int limit) {
    if (limit <= 0)
        return 0;
    std::vector<Entry> entries;
    entries.reserve(m.A.value.size());
    for (int64_t i = 0; i < m.A.rows; ++i)
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
            entries.push_back({i, m.A.index[k], m.A.value[k]});
    std::set<std::pair<std::vector<int64_t>, int64_t>> pool;
    std::set<std::string> names(m.row_names.begin(), m.row_names.end());
    int added = 0;
    auto append = [&](std::vector<int64_t> columns, int64_t rhs) {
        std::sort(columns.begin(), columns.end());
        if (added >= limit || columns.empty() || !pool.insert({columns, rhs}).second)
            return;
        auto row = int64_t(m.rl.size());
        std::string name = "vantage_cut_" + std::to_string(row);
        while (names.count(name))
            name += "_";
        names.insert(name);
        m.row_names.push_back(name);
        m.rl.push_back(-inf);
        m.ru.push_back(double(rhs));
        for (auto j : columns)
            entries.push_back({row, j, 1});
        ++added;
    };
    for (int64_t row = 0; row < m.A.rows && added < limit; ++row) {
        double rhs = m.ru[row];
        if (!std::isfinite(rhs) || rhs < 0 || rhs > 0x1p52 || rhs != std::floor(rhs))
            continue;
        std::vector<std::pair<uint64_t, int64_t>> terms;
        std::set<int64_t> seen;
        bool eligible = true;
        for (auto k = m.A.ptr[row]; k < m.A.ptr[row + 1]; ++k) {
            auto j = m.A.index[k];
            double a = m.A.value[k];
            if (m.types[j] != VarType::Binary || a <= 0 || a > 0x1p52 || a != std::floor(a) ||
                !seen.insert(j).second) {
                eligible = false;
                break;
            }
            terms.emplace_back(uint64_t(a), j);
        }
        if (!eligible || terms.empty())
            continue;
        std::sort(terms.rbegin(), terms.rend());
        uint64_t capacity = uint64_t(rhs), sum = 0;
        std::vector<std::pair<uint64_t, int64_t>> cover;
        for (auto term : terms) {
            cover.push_back(term);
            sum += term.first;
            if (sum > capacity)
                break; // sum cannot exceed 2^53 here.
        }
        if (sum > capacity) {
            // Exact integer arithmetic establishes validity, without a tolerance.
            for (size_t k = cover.size(); k-- > 0;)
                if (sum - cover[k].first > capacity) {
                    sum -= cover[k].first;
                    cover.erase(cover.begin() + k);
                }
            std::vector<int64_t> columns;
            for (auto term : cover)
                columns.push_back(term.second);
            append(columns, int64_t(columns.size()) - 1);
        }
        // A descending prefix is a clique if its two smallest coefficients
        // already exceed capacity: every pair is then mutually incompatible.
        std::vector<int64_t> clique{terms[0].second};
        for (size_t k = 1; k < terms.size(); ++k) {
            if (terms[k - 1].first + terms[k].first <= capacity)
                break;
            clique.push_back(terms[k].second);
        }
        if (clique.size() >= 2)
            append(clique, 1);
    }
    if (added) {
        m.A = Sparse::build(m.rl.size(), m.c.size(), std::move(entries));
        m.validate();
    }
    return added;
}

Model distance_projection_model(const Model &m, const std::vector<double> &target) {
    if (target.size() != m.c.size() || m.is_qp())
        throw std::runtime_error("Invalid feasibility-pump projection");
    Model projection = m;
    projection.name = m.name + "_distance";
    projection.offset = 0;
    projection.sense = 1;
    std::fill(projection.c.begin(), projection.c.end(), 0);
    std::fill(projection.types.begin(), projection.types.end(), VarType::Continuous);
    std::vector<Entry> entries;
    for (int64_t i = 0; i < m.A.rows; ++i)
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
            entries.push_back({i, m.A.index[k], m.A.value[k]});
    std::set<std::string> names(m.names.begin(), m.names.end());
    std::set<std::string> rows(m.row_names.begin(), m.row_names.end());
    for (size_t j = 0; j < m.c.size(); ++j) {
        if (m.types[j] == VarType::Continuous)
            continue;
        if (!std::isfinite(target[j]))
            throw std::runtime_error("Nonfinite pump target");
        auto d = int64_t(projection.c.size());
        std::string name = "vantage_distance_" + std::to_string(j);
        while (names.count(name))
            name += "_";
        names.insert(name);
        projection.names.push_back(name);
        projection.c.push_back(1);
        projection.q.push_back(0);
        projection.lb.push_back(0);
        projection.ub.push_back(inf);
        projection.types.push_back(VarType::Continuous);
        for (int sign : {-1, 1}) {
            auto row = int64_t(projection.rl.size());
            std::string rowname = "vantage_distance_row_" + std::to_string(row);
            while (rows.count(rowname))
                rowname += "_";
            rows.insert(rowname);
            projection.row_names.push_back(rowname);
            projection.rl.push_back(-inf);
            projection.ru.push_back(sign * target[j]);
            entries.push_back({row, int64_t(j), double(sign)});
            entries.push_back({row, d, -1});
        }
    }
    projection.A = Sparse::build(projection.rl.size(), projection.c.size(), std::move(entries));
    projection.validate();
    return projection;
}
} // namespace vantage
