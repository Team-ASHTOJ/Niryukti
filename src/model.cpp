#include "vantage/vantage.hpp"
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
namespace vantage {
Sparse Sparse::build(int64_t m, int64_t n, std::vector<Entry> e) {
    if (m < 0 || n < 0)
        throw std::runtime_error("Negative matrix dimension");
    for (auto a : e)
        if (a.row < 0 || a.row >= m || a.col < 0 || a.col >= n || !std::isfinite(a.value))
            throw std::runtime_error("Invalid sparse entry");
    std::sort(e.begin(), e.end(),
              [](auto a, auto b) { return a.row < b.row || (a.row == b.row && a.col < b.col); });
    Sparse s;
    s.rows = m;
    s.cols = n;
    s.ptr.assign(m + 1, 0);
    for (size_t k = 0; k < e.size();) {
        auto a = e[k++];
        while (k < e.size() && e[k].row == a.row && e[k].col == a.col)
            a.value += e[k++].value;
        if (!std::isfinite(a.value))
            throw std::runtime_error("Coefficient sum overflow");
        if (a.value != 0) {
            s.ptr[a.row + 1]++;
            s.index.push_back(a.col);
            s.value.push_back(a.value);
        }
    }
    for (int64_t i = 0; i < m; i++)
        s.ptr[i + 1] += s.ptr[i];
    return s;
}
Sparse Sparse::transpose() const {
    Sparse t;
    t.rows = cols;
    t.cols = rows;
    t.ptr.assign(cols + 1, 0);
    t.index.resize(value.size());
    t.value.resize(value.size());
    for (auto j : index)
        t.ptr[j + 1]++;
    for (int64_t j = 0; j < cols; j++)
        t.ptr[j + 1] += t.ptr[j];
    auto pos = t.ptr;
    for (int64_t i = 0; i < rows; i++)
        for (auto k = ptr[i]; k < ptr[i + 1]; k++) {
            auto p = pos[index[k]]++;
            t.index[p] = i;
            t.value[p] = value[k];
        }
    return t;
}
std::vector<double> Sparse::multiply(const std::vector<double> &x) const {
    if (x.size() != size_t(cols))
        throw std::runtime_error("SpMV dimension mismatch");
    std::vector<double> y(rows);
#ifdef _OPENMP
#pragma omp parallel for if (rows > 10000)
#endif
    for (int64_t i = 0; i < rows; i++) {
        double v = 0;
        for (auto k = ptr[i]; k < ptr[i + 1]; k++)
            v += value[k] * x[index[k]];
        y[i] = v;
    }
    return y;
}
void Model::validate() const {
    auto n = c.size(), m = rl.size();
    if (q.size() != n || lb.size() != n || ub.size() != n || types.size() != n ||
        names.size() != n || ru.size() != m || row_names.size() != m || A.rows != int64_t(m) ||
        A.cols != int64_t(n))
        throw std::runtime_error("Inconsistent model dimensions");
    if (A.ptr.size() != m + 1 || A.ptr.front() != 0 || A.ptr.back() != int64_t(A.value.size()) ||
        A.index.size() != A.value.size())
        throw std::runtime_error("Invalid CSR storage");
    if (!std::isfinite(offset) || (sense != 1 && sense != -1))
        throw std::runtime_error("Invalid objective offset/sense");
    std::unordered_set<std::string> seen;
    for (size_t j = 0; j < n; j++) {
        if (!std::isfinite(c[j]) || !std::isfinite(q[j]))
            throw std::runtime_error("Nonfinite objective coefficient");
        if (std::isnan(lb[j]) || std::isnan(ub[j]) || lb[j] == inf || ub[j] == -inf ||
            lb[j] > ub[j])
            throw std::runtime_error("Invalid bounds for " + names[j]);
        if (!seen.insert(names[j]).second || names[j].empty())
            throw std::runtime_error("Duplicate/empty variable name");
        if (types[j] == VarType::Binary && (lb[j] < 0 || ub[j] > 1))
            throw std::runtime_error("Binary bounds outside [0,1]");
    }
    seen.clear();
    for (size_t i = 0; i < m; i++) {
        if (std::isnan(rl[i]) || std::isnan(ru[i]) || rl[i] == inf || ru[i] == -inf ||
            rl[i] > ru[i])
            throw std::runtime_error("Invalid row bounds");
        if (!seen.insert(row_names[i]).second || row_names[i].empty())
            throw std::runtime_error("Duplicate/empty row name");
        if (A.ptr[i] > A.ptr[i + 1] || A.ptr[i] < 0 || A.ptr[i + 1] > int64_t(A.value.size()))
            throw std::runtime_error("Invalid CSR offsets");
    }
    for (size_t k = 0; k < A.value.size(); k++)
        if (A.index[k] < 0 || A.index[k] >= int64_t(n) || !std::isfinite(A.value[k]))
            throw std::runtime_error("Invalid matrix coefficient/index");
    if (!Q.value.empty()) {
        if (Q.rows != int64_t(n) || Q.cols != int64_t(n) || Q.ptr.size() != n + 1 ||
            Q.ptr.front() != 0 || Q.ptr.back() != int64_t(Q.value.size()) ||
            Q.index.size() != Q.value.size())
            throw std::runtime_error("Invalid quadratic CSR shape");
        for (size_t i = 0; i < n; ++i)
            if (Q.ptr[i] < 0 || Q.ptr[i] > Q.ptr[i + 1] || Q.ptr[i + 1] > int64_t(Q.value.size()))
                throw std::runtime_error("Invalid quadratic CSR offsets");
        for (size_t k = 0; k < Q.value.size(); ++k)
            if (Q.index[k] < 0 || Q.index[k] >= int64_t(n) || !std::isfinite(Q.value[k]))
                throw std::runtime_error("Invalid quadratic coefficient/index");
        auto qt = Q.transpose();
        if (qt.ptr != Q.ptr || qt.index != Q.index || qt.value != Q.value)
            throw std::runtime_error("Quadratic matrix must be exactly symmetric");
        bool dominant = true;
        for (size_t i = 0; i < n; ++i) {
            long double diagonal = q[i], off = 0;
            if (Q.ptr[i] > Q.ptr[i + 1])
                throw std::runtime_error("Invalid quadratic CSR offsets");
            int64_t previous = -1;
            for (auto k = Q.ptr[i]; k < Q.ptr[i + 1]; ++k) {
                auto j = Q.index[k];
                auto v = Q.value[k];
                if (j <= previous || j >= int64_t(n) || !std::isfinite(v))
                    throw std::runtime_error("Invalid quadratic coefficient/index");
                previous = j;
                if (j == int64_t(i))
                    diagonal += v;
                else
                    off += std::abs(v);
            }
            dominant &= diagonal >= off;
        }
        if (!dominant) {
            if (n > 256)
                throw std::runtime_error(
                    "UNSUPPORTED: large sparse Q requires a diagonal-dominance PSD certificate");
            // Dense storage is confined to small-model convexity validation, never iterations.
            std::vector<long double> h(n * n);
            for (size_t i = 0; i < n; ++i) {
                h[i * n + i] = q[i];
                for (auto k = Q.ptr[i]; k < Q.ptr[i + 1]; ++k)
                    h[i * n + Q.index[k]] += Q.value[k];
            }
            for (size_t k = 0; k < n; ++k) {
                auto pivot = h[k * n + k];
                if (pivot < 0 || !std::isfinite(pivot))
                    throw std::runtime_error("UNSUPPORTED: quadratic objective is not PSD");
                if (pivot == 0) {
                    for (size_t i = k + 1; i < n; ++i)
                        if (h[i * n + k] != 0)
                            throw std::runtime_error("UNSUPPORTED: quadratic objective is not PSD");
                    continue;
                }
                for (size_t i = k + 1; i < n; ++i)
                    for (size_t j = i; j < n; ++j) {
                        h[j * n + i] -= h[i * n + k] * h[j * n + k] / pivot;
                        h[i * n + j] = h[j * n + i];
                    }
            }
        }
    }
}
double Model::quadratic_norm_bound() const {
    long double largest = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        long double sum = std::abs(q[i]);
        if (!Q.value.empty())
            for (auto k = Q.ptr[i]; k < Q.ptr[i + 1]; ++k)
                sum += std::abs(Q.value[k]);
        largest = std::max(largest, sum);
    }
    return std::nextafter(double(largest), inf);
}
bool Model::is_mip() const {
    return std::any_of(types.begin(), types.end(), [](auto t) { return t != VarType::Continuous; });
}
bool Model::is_qp() const {
    return !Q.value.empty() || std::any_of(q.begin(), q.end(), [](double v) { return v != 0; });
}
std::string Model::fingerprint() const {
    uint64_t h = 14695981039346656037ULL;
    auto bytes = [&](const auto &v) {
        const auto *p = reinterpret_cast<const unsigned char *>(&v);
        for (size_t i = 0; i < sizeof(v); i++) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    };
    bytes(A.rows);
    bytes(A.cols);
    bytes(offset);
    bytes(sense);
    for (auto *v : {&c, &q, &lb, &ub, &rl, &ru, &A.value})
        for (auto a : *v)
            bytes(a);
    for (auto a : A.ptr)
        bytes(a);
    for (auto a : A.index)
        bytes(a);
    if (!Q.value.empty()) {
        for (auto a : Q.ptr)
            bytes(a);
        for (auto a : Q.index)
            bytes(a);
        for (auto a : Q.value)
            bytes(a);
    }
    for (auto a : types) {
        int t = int(a);
        bytes(t);
    }
    std::ostringstream os;
    os << std::hex << h;
    return os.str();
}
} // namespace vantage
