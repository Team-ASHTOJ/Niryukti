#include "internal.hpp"
#include "json.hpp"
#include "vantage/analysis.hpp"
#include <ctime>
#include <fstream>
#include <map>
#include <sstream>
// Adaptive strategy layer. Each solve can append a JSON-lines record of structural model
// features, the method/backend used and the independently verified outcome. A recommendation
// ranks methods by distance-weighted verified success and runtime over structurally similar
// past solves (k-nearest-neighbour algorithm selection; Rice 1976, Kotthoff 2016).
namespace vantage {
namespace {
using json = nlohmann::json;
constexpr int feature_count = 7;
const char *feature_names[feature_count] = {"log_rows",          "log_columns",
                                            "log_nonzeros",      "integer_fraction",
                                            "quadratic",         "log_coefficient_range",
                                            "log_row_degree"};
std::vector<double> features(const Model &m) {
    auto advice = advise_model(m, Options{}, Hardware{});
    double n = double(m.c.size()), rows = double(m.A.rows), nnz = double(m.A.value.size());
    double integers = 0;
    for (auto t : m.types)
        integers += t != VarType::Continuous;
    return {std::log10(1 + rows),
            std::log10(1 + n),
            std::log10(1 + nnz),
            n > 0 ? integers / n : 0.,
            m.is_qp() ? 1. : 0.,
            std::log10(std::max(1., advice.coefficient_range)),
            std::log10(1 + (rows > 0 ? nnz / rows : 0.))};
}
std::string family(const std::string &selected) {
    for (auto name : {"dual-simplex", "concurrent", "barrier", "r2hpdhg", "rhpdhg", "halpern"})
        if (selected.find(name) != std::string::npos)
            return name;
    if (selected.find("simplex") != std::string::npos)
        return "simplex";
    if (selected.find("pdhg") != std::string::npos || selected == "smooth-primal-dual")
        return "pdhg";
    return "";
}
struct Record {
    std::vector<double> f;
    std::string method, device, fingerprint;
    bool success;
    double seconds;
};
std::vector<Record> load(const std::string &path) {
    std::vector<Record> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        try {
            auto j = json::parse(line);
            Record r;
            r.f = j.at("features").get<std::vector<double>>();
            if (r.f.size() != feature_count)
                continue;
            r.method = j.at("method").get<std::string>();
            r.device = j.value("device", "cpu");
            r.fingerprint = j.value("fingerprint", "");
            r.success = j.value("verified_success", false);
            r.seconds = std::max(1e-6, j.value("seconds", 1e9));
            if (!r.method.empty())
                out.push_back(r);
        } catch (...) {
            // Skip malformed lines; history is advisory and append-only.
        }
    }
    return out;
}
struct Ranking {
    std::string method, device;
    double weight = 0, success = 0, log_time = 0, time_weight = 0, score = inf;
    int64_t runs = 0, successes = 0;
};
std::vector<Ranking> rank(const std::vector<Record> &records, const Model &m, int64_t &neighbours) {
    auto q = features(m);
    auto fp = m.fingerprint();
    std::map<std::string, Ranking> by;
    neighbours = 0;
    for (auto &r : records) {
        double d2 = 0;
        for (int k = 0; k < feature_count; ++k)
            d2 += (r.f[k] - q[k]) * (r.f[k] - q[k]);
        double distance = r.fingerprint == fp ? 0 : std::sqrt(d2);
        if (distance > 1.5)
            continue;
        ++neighbours;
        double w = 1 / (0.1 + distance);
        auto &g = by[r.method + "@" + r.device];
        g.method = r.method;
        g.device = r.device;
        g.weight += w;
        g.runs++;
        if (r.success) {
            g.success += w;
            g.successes++;
            g.log_time += w * std::log10(r.seconds);
            g.time_weight += w;
        }
    }
    std::vector<Ranking> out;
    for (auto &[key, g] : by) {
        double rate = g.success / g.weight;
        double time = g.time_weight > 0 ? g.log_time / g.time_weight : 3;
        // A failure costs as much as three decades of runtime.
        g.score = time + 3 * (1 - rate);
        out.push_back(g);
    }
    std::sort(out.begin(), out.end(), [](auto &a, auto &b) { return a.score < b.score; });
    return out;
}
} // namespace

void record_history(const std::string &path, const Model &m, const Options &o, const Result &r) {
    auto selected = r.relaxation_method_selected.empty() ? r.method_selected
                                                         : r.relaxation_method_selected;
    std::string method = o.method != "auto" && o.method != "learned" ? o.method : family(selected);
    bool verified = r.accuracy.finite && r.accuracy.primal <= 1e-6 &&
                    r.accuracy.integrality <= 1e-6;
    json j = {{"version", 1},
              {"time", int64_t(std::time(nullptr))},
              {"model", m.name},
              {"fingerprint", m.fingerprint()},
              {"features", features(m)},
              {"method", method},
              {"method_selected", r.method_selected},
              {"device", r.backend},
              {"status", r.status},
              {"verified_success", r.status == "OPTIMAL" && verified},
              {"seconds", r.seconds},
              {"iterations", r.iterations},
              {"nodes", r.nodes}};
    std::ofstream out(path, std::ios::app);
    if (!out)
        throw std::runtime_error("Cannot append solve history to " + path);
    out << j.dump() << '\n';
}

std::string learned_method(const std::string &path, const Model &m, std::string *reason) {
    auto records = load(path);
    int64_t neighbours = 0;
    auto ranked = rank(records, m, neighbours);
    for (auto &g : ranked)
        if (g.runs >= 2 && g.successes >= 1 && neighbours >= 3) {
            if (reason) {
                std::ostringstream s;
                s << "Learned from " << neighbours << " similar solves: " << g.method << " on "
                  << g.device << " succeeded " << g.successes << "/" << g.runs
                  << " times, typical " << std::pow(10, g.log_time / g.time_weight) << " s";
                *reason = s.str();
            }
            return g.method;
        }
    if (reason)
        *reason = "Insufficient similar history (" + std::to_string(neighbours) +
                  " neighbours); structural advisor used";
    return "";
}

std::string recommend_json(const std::string &path, const Model &m, const Options &o) {
    auto records = load(path);
    int64_t neighbours = 0;
    auto ranked = rank(records, m, neighbours);
    std::string reason;
    auto learned = learned_method(path, m, &reason);
    auto f = features(m);
    json feats = json::object();
    for (int k = 0; k < feature_count; ++k)
        feats[feature_names[k]] = f[k];
    json table = json::array();
    for (auto &g : ranked)
        table.push_back({{"method", g.method},
                         {"device", g.device},
                         {"runs", g.runs},
                         {"verified_successes", g.successes},
                         {"weighted_success_rate", g.success / g.weight},
                         {"typical_seconds", g.time_weight > 0
                                                 ? json(std::pow(10, g.log_time / g.time_weight))
                                                 : json()},
                         {"score", g.score}});
    Options automatic = o;
    automatic.method = "auto";
    auto advice = advise_model(m, automatic, hardware_info());
    return json({{"analysis", "LEARNED_STRATEGY"},
                 {"history", path},
                 {"records", records.size()},
                 {"similar_records", neighbours},
                 {"features", feats},
                 {"ranking", table},
                 {"recommended_method", learned.empty() ? advice.method : learned},
                 {"source", learned.empty() ? "structural-advisor" : "history"},
                 {"reason", learned.empty() ? reason + "; " + advice.reason : reason},
                 {"scope", "Advisory ranking from local verified outcomes; not a runtime "
                           "guarantee"}})
        .dump(2);
}
} // namespace vantage
