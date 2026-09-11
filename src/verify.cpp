#include "vantage/vantage.hpp"
namespace vantage {
namespace {
// Enclose each elementary LP dual-bound operation with outward-rounded long
// doubles. Infinite-box incompatibility yields -infinity, never a guessed bound.
long double down(long double v) {
    return std::nextafter(v, -std::numeric_limits<long double>::infinity());
}
long double up(long double v) {
    return std::nextafter(v, std::numeric_limits<long double>::infinity());
}
double lp_lower_bound(const Model &m, const Sparse &at, const std::vector<double> &y,
                      bool feasibility = false) {
    long double lower = feasibility ? 0 : m.offset;
    for (size_t i = 0; i < y.size(); i++)
        if (y[i] != 0) {
            double b = y[i] > 0 ? m.ru[i] : m.rl[i];
            if (!std::isfinite(b))
                return -inf;
            if (b == 0)
                continue;
            long double term = (long double)y[i] * b;
            lower = down(lower - up(term));
        }
    for (size_t j = 0; j < m.c.size(); j++) {
        long double gl = feasibility ? 0 : m.c[j], gu = gl;
        for (auto k = at.ptr[j]; k < at.ptr[j + 1]; k++)
            if (y[at.index[k]] != 0 && at.value[k] != 0) {
                long double v = (long double)at.value[k] * y[at.index[k]];
                gl = down(gl + down(v));
                gu = up(gu + up(v));
            }
        if ((!std::isfinite(m.lb[j]) && gu > 0) || (!std::isfinite(m.ub[j]) && gl < 0))
            return -inf;
        long double term = std::numeric_limits<long double>::infinity();
        if (gl <= 0 && gu >= 0)
            term = 0;
        for (double b : {m.lb[j], m.ub[j]})
            if (std::isfinite(b)) {
                term = std::min(term, (b == 0 || gl == 0) ? 0 : down(gl * b));
                term = std::min(term, (b == 0 || gu == 0) ? 0 : down(gu * b));
            }
        if (gl == 0 && gu == 0)
            term = 0;
        if (!std::isfinite(term))
            return -inf;
        if (term != 0)
            lower = down(lower + term);
    }
    double result = double(lower);
    if (!std::isfinite(result))
        return -inf;
    return std::nextafter(result, -inf);
}
} // namespace
Verifier::Verifier(const Model &source)
    : model(source), transpose(source.A.transpose()), activity(source.A.rows),
      transpose_product(source.A.cols) {}

double Verifier::infeasibility_bound(const std::vector<double> &y) const {
    if (y.size() != model.rl.size())
        return -inf;
    for (auto v : y)
        if (!std::isfinite(v))
            return -inf;
    return lp_lower_bound(model, transpose, y, true);
}

Accuracy Verifier::evaluate(const std::vector<double> &x, const std::vector<double> &y,
                            bool compute_safe_bound) {
    const auto &m = model;
    Accuracy a;
    if (x.size() != m.c.size() || y.size() != m.rl.size())
        return a;
    for (auto v : x)
        if (!std::isfinite(v))
            return a;
    for (auto v : y)
        if (!std::isfinite(v))
            return a;
    auto multiply_into = [](const Sparse &a, const std::vector<double> &input,
                            std::vector<double> &output) {
#ifdef _OPENMP
#pragma omp parallel for if (a.rows > 10000)
#endif
        for (int64_t i = 0; i < a.rows; i++) {
            double sum = 0;
            for (auto k = a.ptr[i]; k < a.ptr[i + 1]; k++)
                sum += a.value[k] * input[a.index[k]];
            output[i] = sum;
        }
    };
    multiply_into(m.A, x, activity);
    multiply_into(transpose, y, transpose_product);
    const auto &ax = activity;
    const auto &aty = transpose_product;
    for (auto v : ax)
        if (!std::isfinite(v))
            return Accuracy{};
    for (auto v : aty)
        if (!std::isfinite(v))
            return Accuracy{};
    a.objective = m.offset;
    a.primal = 0;
    a.dual = 0;
    a.integrality = 0;
    a.primal_absolute = 0;
    a.dual_absolute = 0;
    long double support = 0, box_min = 0, complement = 0;
    bool bound_ok = true;
    // Per-row normalization prevents an unrelated enormous RHS from hiding a violation.
    for (size_t i = 0; i < y.size(); i++) {
        double v = std::max({0., m.rl[i] - ax[i], ax[i] - m.ru[i]});
        if (std::isfinite(m.rl[i]))
            a.primal = std::max(a.primal, std::max(0., m.rl[i] - ax[i]) / (1 + std::abs(m.rl[i])));
        if (std::isfinite(m.ru[i]))
            a.primal = std::max(a.primal, std::max(0., ax[i] - m.ru[i]) / (1 + std::abs(m.ru[i])));
        a.primal_absolute = std::max(a.primal_absolute, v);
        double endpoint = y[i] > 0 ? m.ru[i] : m.rl[i];
        if (y[i] != 0) {
            if (std::isfinite(endpoint)) {
                support += (long double)y[i] * endpoint;
                complement += std::abs((long double)y[i] * (endpoint - ax[i]));
            } else {
                bound_ok = false;
                a.dual_absolute = std::max(a.dual_absolute, std::abs(y[i]));
                a.dual = std::max(a.dual, std::abs(y[i]));
            }
        }
    }
    double cscale = 1;
    for (auto v : m.c)
        cscale = std::max(cscale, 1 + std::abs(v));
    for (size_t j = 0; j < x.size(); j++) {
        double v = std::max({0., m.lb[j] - x[j], x[j] - m.ub[j]});
        double bs = 1 + std::abs(x[j]);
        a.primal = std::max(a.primal, v / bs);
        a.primal_absolute = std::max(a.primal_absolute, v);
        if (m.types[j] != VarType::Continuous)
            a.integrality = std::max(a.integrality, std::abs(x[j] - std::round(x[j])));
        a.objective += m.c[j] * x[j] + .5 * m.q[j] * x[j] * x[j];
        double linear = m.c[j] + aty[j], g = linear + m.q[j] * x[j];
        if (!std::isfinite(linear) || !std::isfinite(g))
            return Accuracy{};
        // Box normal-cone stationarity via the projected-gradient mapping.
        double d = std::clamp(g, x[j] - m.ub[j], x[j] - m.lb[j]);
        a.dual_absolute = std::max(a.dual_absolute, std::abs(d));
        a.dual = std::max(a.dual, std::abs(d) / cscale);
        if (g > 0 && std::isfinite(m.lb[j]))
            complement += std::abs((long double)g * (x[j] - m.lb[j]));
        if (g < 0 && std::isfinite(m.ub[j]))
            complement += std::abs((long double)g * (m.ub[j] - x[j]));
        // Independent separable minimization of the Lagrangian over the ORIGINAL box.
        double arg = 0;
        if (m.q[j] > 0)
            arg = std::clamp(-linear / m.q[j], m.lb[j], m.ub[j]);
        else if (linear > 0)
            arg = m.lb[j];
        else if (linear < 0)
            arg = m.ub[j];
        else
            arg = std::clamp(0., m.lb[j], m.ub[j]);
        if (!std::isfinite(arg)) {
            bound_ok = false;
        } else
            box_min += (long double)linear * arg + .5L * m.q[j] * arg * arg;
    }
    a.complementarity = double(complement);
    a.gap = a.complementarity / (1 + std::abs(a.objective - m.offset));
    if (bound_ok) {
        long double lower = (long double)m.offset - support +
                            box_min; // Guard accumulation roundoff; this is numerical, not
                                     // interval-arithmetic certification.
        long double guard =
            64 * std::numeric_limits<double>::epsilon() *
            (1 + std::abs((long double)m.offset) + std::abs(support) + std::abs(box_min));
        a.lower_bound = double(lower - guard);
        a.gap = std::max(a.gap, std::abs(a.objective - a.lower_bound) /
                                    (1 + std::abs(a.objective) + std::abs(a.lower_bound)));
    }
    if (!m.is_qp())
        a.lower_bound = compute_safe_bound ? lp_lower_bound(m, transpose, y) : -inf;
    if (std::isfinite(a.lower_bound))
        a.gap = std::max(a.gap, std::abs(a.objective - a.lower_bound) /
                                    (1 + std::abs(a.objective) + std::abs(a.lower_bound)));
    a.kkt = std::max({a.primal, a.dual, a.gap});
    a.finite = std::isfinite(a.objective) && std::isfinite(a.kkt) && std::isfinite(a.integrality) &&
               std::isfinite(a.gap) && std::isfinite(a.primal) && std::isfinite(a.dual);
    if (!a.finite) {
        a.kkt = inf;
        a.lower_bound = -inf;
    }
    return a;
}
Accuracy verify(const Model &m, const std::vector<double> &x, const std::vector<double> &y) {
    Verifier verifier(m);
    return verifier.evaluate(x, y);
}
} // namespace vantage
