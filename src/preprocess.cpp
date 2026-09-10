#include "internal.hpp"
#include <numeric>
namespace vantage {
std::vector<double> Prepared::restore_x(const std::vector<double> &x) const {
    auto out = fixed;
    for (size_t j = 0; j < cols.size(); j++)
        out[cols[j]] = x[j] * column_scale[j];
    return out;
}
std::vector<double> Prepared::restore_y(const std::vector<double> &y, size_t n) const {
    std::vector<double> out(n);
    for (size_t i = 0; i < rows.size(); i++)
        out[rows[i]] = y[i] * row_scale[i] * objective_scale;
    return out;
}
Prepared prepare(const Model &src, const Options &o) {
    Prepared p;
    p.fixed.assign(src.c.size(), 0);
    Model &m = p.model;
    m.name = src.name;
    m.offset = src.offset;
    m.sense = src.sense;
    std::vector<int64_t> map(src.c.size(), -1);
    std::vector<bool> used(src.c.size(), false);
    for (auto j : src.A.index)
        used[j] = true;
    for (size_t j = 0; j < src.c.size(); j++) {
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
    std::vector<Entry> e;
    for (size_t i = 0; i < src.rl.size(); i++) {
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
        auto row = p.rows.size();
        p.rows.push_back(i);
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
    // Ruiz infinity-norm equilibration, x_original = D_c x_scaled.
    for (int pass = 0; pass < o.scaling_passes; pass++) {
        for (size_t i = 0; i < m.rl.size(); i++) {
            double norm = 0;
            for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++)
                norm = std::max(norm, std::abs(m.A.value[k]));
            double f = norm > 0 ? std::clamp(1 / std::sqrt(norm), 1e-3, 1e3) : 1;
            p.row_scale[i] *= f;
            m.rl[i] *= f;
            m.ru[i] *= f;
            for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++)
                m.A.value[k] *= f;
        }
        std::vector<double> norm(m.c.size());
        for (size_t k = 0; k < m.A.value.size(); k++)
            norm[m.A.index[k]] = std::max(norm[m.A.index[k]], std::abs(m.A.value[k]));
        for (size_t j = 0; j < norm.size(); j++) {
            double f = norm[j] > 0 ? std::clamp(1 / std::sqrt(norm[j]), 1e-3, 1e3) : 1;
            norm[j] = f;
            p.column_scale[j] *= f;
            m.c[j] *= f;
            m.q[j] *= f * f;
            m.lb[j] /= f;
            m.ub[j] /= f;
        }
        for (size_t k = 0; k < m.A.value.size(); k++)
            m.A.value[k] *= norm[m.A.index[k]];
    }
    if (o.scaling_passes > 0) {
        double cnorm = 0;
        for (auto c : m.c)
            cnorm = std::max(cnorm, std::abs(c));
        for (auto q : m.q)
            cnorm = std::max(cnorm, std::abs(q));
        p.objective_scale = cnorm > 0 ? cnorm : 1;
        for (auto &c : m.c)
            c /= p.objective_scale;
        for (auto &q : m.q)
            q /= p.objective_scale;
        m.offset /= p.objective_scale;
    }
    m.validate();
    return p;
}
} // namespace vantage
