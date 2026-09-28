#include "internal.hpp"
#include <map>
#include <numeric>
namespace vantage {
namespace {
bool representable(double before, double after) {
    return std::isfinite(before) ? std::isfinite(after) && (before == 0 || after != 0)
                                 : before == after;
}
// Optional preconditioners never erase a nonzero or turn a finite bound into infinity.
// Skip unsafe row/column transformations as a whole to preserve the model mapping.
void extra_scale(Prepared &p, std::vector<double> row, std::vector<double> col) {
    auto &m = p.model;
    for (size_t i = 0; i < row.size(); i++) {
        double f = row[i];
        bool safe = representable(p.row_scale[i], p.row_scale[i] * f) &&
                    representable(m.rl[i], m.rl[i] * f) && representable(m.ru[i], m.ru[i] * f);
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++)
            safe &= representable(m.A.value[k], m.A.value[k] * f);
        if (!safe)
            continue;
        p.row_scale[i] *= f;
        m.rl[i] *= f;
        m.ru[i] *= f;
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++)
            m.A.value[k] *= f;
    }
    for (size_t j = 0; j < col.size(); j++) {
        double f = col[j];
        if (!representable(p.column_scale[j], p.column_scale[j] * f) ||
            !representable(m.c[j], m.c[j] * f) || !representable(m.q[j], m.q[j] * (f * f)) ||
            !representable(m.lb[j], m.lb[j] / f) || !representable(m.ub[j], m.ub[j] / f))
            col[j] = 1;
    }
    for (size_t k = 0; k < m.A.value.size(); k++) {
        auto j = m.A.index[k];
        if (!representable(m.A.value[k], m.A.value[k] * col[j]))
            col[j] = 1;
    }
#ifndef VANTAGE_SPARSE_LU
    // The portable validator recognizes large Q by dominance; avoid breaking that
    // sufficient certificate. Sparse factorization builds support congruent scaling.
    if (!m.Q.value.empty() && m.c.size() > 256)
        std::fill(col.begin(), col.end(), 1);
#endif
    bool qsafe = true;
    for (int64_t i = 0; i < m.Q.rows; ++i)
        for (auto k = m.Q.ptr[i]; k < m.Q.ptr[i + 1]; ++k)
            qsafe &= representable(m.Q.value[k], (m.Q.value[k] * col[i]) * col[m.Q.index[k]]);
    if (!qsafe)
        std::fill(col.begin(), col.end(), 1);
    for (int64_t i = 0; i < m.Q.rows; ++i)
        for (auto k = m.Q.ptr[i]; k < m.Q.ptr[i + 1]; ++k)
            m.Q.value[k] *= col[i] * col[m.Q.index[k]];
    for (size_t j = 0; j < col.size(); j++) {
        double f = col[j];
        p.column_scale[j] *= f;
        m.c[j] *= f;
        m.q[j] *= f * f;
        m.lb[j] /= f;
        m.ub[j] /= f;
    }
    for (size_t k = 0; k < m.A.value.size(); k++)
        m.A.value[k] *= col[m.A.index[k]];
}
double geometric_factor(double low, double high) {
    if (high == 0)
        return 1;
    // Logarithms avoid overflow/underflow of low * high.
    return std::exp(
        std::clamp(-.5 * (std::log(low) + std::log(high)), std::log(1e-3), std::log(1e3)));
}
void geometric_scale(Prepared &p) {
    auto &m = p.model;
    std::vector<double> row(m.rl.size(), 1), col(m.c.size(), 1);
    for (size_t i = 0; i < row.size(); i++) {
        double low = inf, high = 0;
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++) {
            double v = std::abs(m.A.value[k]);
            if (v > 0) {
                low = std::min(low, v);
                high = std::max(high, v);
            }
        }
        row[i] = geometric_factor(low, high);
    }
    extra_scale(p, row, col);
    std::vector<double> low(col.size(), inf), high(col.size());
    for (size_t k = 0; k < m.A.value.size(); k++) {
        auto j = m.A.index[k];
        double v = std::abs(m.A.value[k]);
        if (v > 0) {
            low[j] = std::min(low[j], v);
            high[j] = std::max(high[j], v);
        }
    }
    for (size_t j = 0; j < col.size(); j++)
        col[j] = geometric_factor(low[j], high[j]);
    std::fill(row.begin(), row.end(), 1);
    extra_scale(p, row, col);
}
void pock_chambolle_scale(Prepared &p) {
    auto &m = p.model;
    std::vector<long double> rows(m.rl.size()), cols(m.c.size());
    for (size_t i = 0; i < rows.size(); i++)
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++) {
            long double v = std::abs(m.A.value[k]);
            rows[i] += v;
            cols[m.A.index[k]] += v;
        }
    auto factor = [](long double sum) {
        return sum > 0 && std::isfinite(sum) ? double(std::clamp(1 / std::sqrt(sum), 1e-3L, 1e3L))
                                             : 1.;
    };
    std::vector<double> row(rows.size()), col(cols.size());
    std::transform(rows.begin(), rows.end(), row.begin(), factor);
    std::transform(cols.begin(), cols.end(), col.begin(), factor);
    extra_scale(p, row, col);
}
} // namespace
std::vector<double> Prepared::restore_x(const std::vector<double> &x) const {
    auto out = fixed;
    for (size_t j = 0; j < cols.size(); j++)
        out[cols[j]] = x[j] * column_scale[j];
    return out;
}
std::vector<double> Prepared::restore_y(const std::vector<double> &y, size_t n) const {
    std::vector<double> out(n);
    for (size_t i = 0; i < rows.size(); i++)
        out[y[i] >= 0 ? upper_rows[i] : lower_rows[i]] += y[i] * row_scale[i] * objective_scale;
    return out;
}
Prepared prepare(const Model &src, const Options &o) {
    Prepared p;
    auto start = Clock::now();
    auto stopped = [&]() {
        if (!stop_requested(o) && elapsed(start) < o.time_limit)
            return false;
        p.failure = stop_requested(o) ? "INTERRUPTED" : "TIME_LIMIT";
        p.reason = "Stopped during presolve/scaling before iteration work";
        return true;
    };
    if (stopped())
        return p;
    p.fixed.assign(src.c.size(), 0);
    Model &m = p.model;
    m.name = src.name;
    m.offset = src.offset;
    m.sense = src.sense;
    std::vector<int64_t> map(src.c.size(), -1);
    std::vector<bool> used(src.c.size(), false);
    for (auto j : src.A.index)
        used[j] = true;
    for (auto j : src.Q.index)
        used[j] = true;
    for (int64_t i = 0; i < src.Q.rows; ++i)
        if (src.Q.ptr[i] != src.Q.ptr[i + 1])
            used[i] = true;
    for (size_t j = 0; j < src.c.size(); j++) {
        if ((j & 1023) == 0 && stopped())
            return p;
        double chosen = src.lb[j];
        bool removable = src.lb[j] == src.ub[j];
        if (!used[j]) {
            chosen = src.q[j] > 0   ? std::clamp(-src.c[j] / src.q[j], src.lb[j], src.ub[j])
                     : src.c[j] > 0 ? src.lb[j]
                     : src.c[j] < 0 ? src.ub[j]
                                    : std::clamp(0., src.lb[j], src.ub[j]);
            removable = std::isfinite(chosen);
        }
        if (o.presolve && removable) {
            p.fixed[j] = chosen;
            m.offset += src.c[j] * chosen + .5 * src.q[j] * chosen * chosen;
        } else {
            map[j] = p.cols.size();
            p.cols.push_back(j);
            m.c.push_back(src.c[j]);
            m.q.push_back(src.q[j]);
            m.lb.push_back(src.lb[j]);
            m.ub.push_back(src.ub[j]);
            m.names.push_back(src.names[j]);
            m.types.push_back(VarType::Continuous);
        }
    }
    std::vector<Entry> qe;
    for (int64_t i = 0; i < src.Q.rows; ++i)
        for (auto k = src.Q.ptr[i]; k < src.Q.ptr[i + 1]; ++k) {
            auto j = src.Q.index[k];
            auto v = src.Q.value[k];
            if (map[i] >= 0 && map[j] >= 0)
                qe.push_back({map[i], map[j], v});
            else if (map[i] >= 0)
                m.c[map[i]] += v * p.fixed[j];
            else if (map[j] < 0)
                m.offset += .5 * v * p.fixed[i] * p.fixed[j];
        }
    m.Q = Sparse::build(m.c.size(), m.c.size(), std::move(qe));
    std::vector<Entry> e;
    std::map<std::vector<std::pair<int64_t, double>>, size_t> duplicate_rows;
    for (size_t i = 0; i < src.rl.size(); i++) {
        if ((i & 1023) == 0 && stopped())
            return p;
        long double shift = 0, shift_magnitude = 0;
        int64_t count = 0;
        for (auto k = src.A.ptr[i]; k < src.A.ptr[i + 1]; k++) {
            auto j = src.A.index[k];
            if (map[j] < 0) {
                long double term = (long double)src.A.value[k] * p.fixed[j];
                shift += term;
                shift_magnitude += std::abs(term);
            } else
                count++;
        }
        double l = src.rl[i] - double(shift), u = src.ru[i] - double(shift);
        if (count == 0 && o.presolve) {
            long double guard = 8 * std::numeric_limits<long double>::epsilon() *
                                (src.A.ptr[i + 1] - src.A.ptr[i] + 1) * (1 + shift_magnitude);
            if (shift + guard < src.rl[i] || shift - guard > src.ru[i]) {
                p.failure = "INFEASIBLE";
                p.reason = "Empty row after fixed-variable substitution: " + src.row_names[i];
                return p;
            }
            if (l <= 0 && u >= 0)
                continue;
        }
        // An activity interval disjoint from the row interval is a direct presolve proof.
        long double amin = 0, amax = 0, magnitude = 0;
        for (auto k = src.A.ptr[i]; k < src.A.ptr[i + 1]; k++) {
            auto j = src.A.index[k];
            double v = src.A.value[k];
            amin += (long double)v * (v > 0 ? src.lb[j] : src.ub[j]);
            amax += (long double)v * (v > 0 ? src.ub[j] : src.lb[j]);
            magnitude += std::abs((long double)v * (v > 0 ? src.lb[j] : src.ub[j])) +
                         std::abs((long double)v * (v > 0 ? src.ub[j] : src.lb[j]));
        }
        if (o.presolve) {
            long double slack = 8 * std::numeric_limits<long double>::epsilon() *
                                (src.A.ptr[i + 1] - src.A.ptr[i] + 1) * (1 + magnitude);
            if (std::isfinite(amin) && std::isfinite(amax) &&
                (amin > src.ru[i] + slack || amax < src.rl[i] - slack)) {
                p.failure = "INFEASIBLE";
                p.reason = "Verified disjoint row activity bounds: " + src.row_names[i];
                return p;
            }
        }
        if (o.presolve && std::isfinite(amin) && std::isfinite(amax)) {
            long double guard = 8 * std::numeric_limits<long double>::epsilon() *
                                (src.A.ptr[i + 1] - src.A.ptr[i] + 1) * (1 + magnitude);
            if (amin - guard >= src.rl[i] && amax + guard <= src.ru[i])
                continue;
        }
        // Exact parallel rows without fixed-variable substitution can be intersected.
        // Restore each dual sign to the original row that supplied its active endpoint.
        if (o.presolve && count == src.A.ptr[i + 1] - src.A.ptr[i]) {
            std::vector<std::pair<int64_t, double>> key;
            for (auto k = src.A.ptr[i]; k < src.A.ptr[i + 1]; ++k)
                key.push_back({map[src.A.index[k]], src.A.value[k]});
            auto found = duplicate_rows.find(key);
            if (found != duplicate_rows.end()) {
                auto r = found->second;
                if (l > m.rl[r]) {
                    m.rl[r] = l;
                    p.lower_rows[r] = i;
                }
                if (u < m.ru[r]) {
                    m.ru[r] = u;
                    p.upper_rows[r] = i;
                }
                if (m.rl[r] > m.ru[r]) {
                    p.failure = "INFEASIBLE";
                    p.reason = "Disjoint bounds on identical constraint rows";
                    p.failure_ray.assign(src.rl.size(), 0);
                    p.failure_ray[p.lower_rows[r]] = -1;
                    p.failure_ray[p.upper_rows[r]] += 1;
                    if (!(Verifier(src).infeasibility_bound(p.failure_ray) > 0))
                        p.failure_ray.clear();
                    return p;
                }
                continue;
            }
            duplicate_rows.emplace(std::move(key), p.rows.size());
        }
        auto row = p.rows.size();
        p.rows.push_back(i);
        p.lower_rows.push_back(i);
        p.upper_rows.push_back(i);
        m.rl.push_back(l);
        m.ru.push_back(u);
        m.row_names.push_back(src.row_names[i]);
        for (auto k = src.A.ptr[i]; k < src.A.ptr[i + 1]; k++)
            if (map[src.A.index[k]] >= 0)
                e.push_back({int64_t(row), map[src.A.index[k]], src.A.value[k]});
    }
    m.A = Sparse::build(m.rl.size(), m.c.size(), std::move(e));
    p.column_scale.assign(m.c.size(), 1);
    p.row_scale.assign(m.rl.size(), 1);
    if (o.scaling == "combined" && o.scaling_passes > 0)
        geometric_scale(p);
    // Ruiz infinity-norm equilibration, x_original = D_c x_scaled.
    for (int pass = 0; pass < o.scaling_passes; pass++) {
        if (stopped())
            return p;
        std::vector<double> row(m.rl.size(), 1), col(m.c.size(), 1);
        for (size_t i = 0; i < row.size(); ++i) {
            double norm = 0;
            for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; ++k)
                norm = std::max(norm, std::abs(m.A.value[k]));
            row[i] = norm > 0 ? std::clamp(1 / std::sqrt(norm), 1e-3, 1e3) : 1.;
        }
        extra_scale(p, row, col);
        std::vector<double> norm(col.size());
        for (size_t k = 0; k < m.A.value.size(); ++k)
            norm[m.A.index[k]] = std::max(norm[m.A.index[k]], std::abs(m.A.value[k]));
        for (size_t j = 0; j < col.size(); ++j)
            col[j] = norm[j] > 0 ? std::clamp(1 / std::sqrt(norm[j]), 1e-3, 1e3) : 1.;
        std::fill(row.begin(), row.end(), 1);
        extra_scale(p, row, col);
    }
    if (o.scaling == "combined" && o.scaling_passes > 0)
        pock_chambolle_scale(p);
    if (o.scaling_passes > 0) {
        double cnorm = 0;
        for (auto c : m.c)
            cnorm = std::max(cnorm, std::abs(c));
        for (auto q : m.q)
            cnorm = std::max(cnorm, std::abs(q));
        for (auto q : m.Q.value)
            cnorm = std::max(cnorm, std::abs(q));
        p.objective_scale = cnorm > 0 ? cnorm : 1;
        for (auto &c : m.c)
            c /= p.objective_scale;
        for (auto &q : m.q)
            q /= p.objective_scale;
        for (auto &q : m.Q.value)
            q /= p.objective_scale;
        m.offset /= p.objective_scale;
    }
    m.validate();
    return p;
}
} // namespace vantage
