#include "internal.hpp"
#include <sstream>
namespace vantage {
Advice advise_model(const Model &m, const Options &o, const Hardware &h) {
    Advice a;
    a.nonzeros = m.A.value.size() + m.Q.value.size();
    a.estimated_gpu_bytes = 48. * a.nonzeros + 240. * (m.A.rows + m.A.cols + 2);
    long double smallest = inf, largest = 0;
    for (double v : m.A.value) {
        if (v != 0) smallest = std::min(smallest, std::abs((long double)v));
        largest = std::max(largest, std::abs((long double)v));
    }
    a.coefficient_range = largest && smallest != inf ? double(largest / smallest) : 1;
    for (size_t j = 0; j < m.c.size(); ++j) {
        a.integer_variables += m.types[j] != VarType::Continuous;
        if (m.lb[j] == m.ub[j]) continue;
        a.transformed_rows += std::isfinite(m.lb[j]);
        a.transformed_rows += std::isfinite(m.ub[j]);
    }
    for (size_t i = 0; i < m.rl.size(); ++i) {
        a.transformed_rows += std::isfinite(m.rl[i]);
        a.transformed_rows += std::isfinite(m.ru[i]) && m.rl[i] != m.ru[i];
        auto degree = m.A.ptr[i + 1] - m.A.ptr[i];
        a.newton_fill_estimate += double(degree) * double(degree);
    }
    bool forced_gpu = o.cuda_graphs || o.matrix_precision == "mixed" || o.gpu_presolve ||
                      o.batch_strong_branching;
    bool fits = h.gpu_available && h.free_gpu_bytes > 0 &&
                a.estimated_gpu_bytes <= .8 * h.free_gpu_bytes;
    a.device = o.device;
    if (o.device == "auto") {
        a.device = fits && (a.nonzeros >= 100000 || forced_gpu) ? gpu_backend_name() : "cpu";
        a.reason = !h.gpu_available ? "CPU: no supported GPU available"
                   : !fits ? "CPU: estimated device storage exceeds the available-memory guard"
                   : a.device == "cpu" ? "CPU: small sparse workload; transfer/launch costs dominate"
                   : "GPU: large sparse workload fits the available-memory guard";
    } else a.reason = "Explicit backend request";
    a.method = o.method;
    if (o.method != "auto") return a;
    a.method = "pdhg";
    if (!o.checkpoint_path.empty() || !o.resume_path.empty() || o.power_iterations > 0 ||
        o.polishing || o.primal_weight == "pid" || !o.adaptive)
        a.reason += "; PDHG preserves checkpoints and requested first-order controls";
    else if (a.device == "cpu" && !m.is_qp() &&
             a.transformed_rows <= simplex_row_limit() && m.c.size() <= 50000 &&
             a.nonzeros <= 300000) {
        a.method = o.initial_basis.empty() ? "simplex" : "dual-simplex";
        a.reason += o.initial_basis.empty() ? "; compact LP admits sparse basis factorization"
                                          : "; compatible basis can accelerate reoptimization";
    } else if (a.device == "cpu" && m.is_qp() && !m.Q.value.empty() && o.tol <= 1e-7 &&
               m.c.size() <= 512 && m.rl.size() <= 512 && a.newton_fill_estimate <= 1000000) {
#ifdef VANTAGE_SPARSE_LU
        a.method = "barrier";
        a.reason += "; small coupled QP with strict tolerance admits sparse Newton steps";
#endif
    } else a.reason += "; matrix-free first-order iterations avoid basis/Newton fill";
    return a;
}
} // namespace vantage
