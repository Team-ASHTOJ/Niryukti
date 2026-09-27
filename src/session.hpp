#pragma once
#include "json.hpp"
#include "vantage/vantage.hpp"
namespace vantage {
class Session {
    Model model;
    Result previous;
    using json = nlohmann::json;
public:
    explicit Session(const std::string &path) : model(read_model(path)) {}
    std::string fingerprint() const { return model.fingerprint(); }
    json request(const json &request) {
            auto action = request.at("action").get<std::string>();
            const std::vector<std::string> allowed = action == "update"
                ? std::vector<std::string>{"action","rows","variables","objective"}
                : std::vector<std::string>{"action","device","method","time_limit","tol","threads","iterations","warm_start"};
            for (auto it=request.begin();it!=request.end();++it)
                if (std::find(allowed.begin(),allowed.end(),it.key())==allowed.end())
                    throw std::runtime_error("Unknown session setting: " + it.key());
            
            if (action == "update") {
                Model candidate = model;
                for (auto section : {"rows", "variables", "objective"}) {
                    if (!request.contains(section)) continue;
                    if (!request[section].is_object()) throw std::runtime_error("Updates must be named objects");
                    const auto &names = std::string(section) == "rows" ? model.row_names : model.names;
                    for (auto it = request[section].begin(); it != request[section].end(); ++it) {
                        auto found = std::find(names.begin(), names.end(), it.key());
                        if (found == names.end()) throw std::runtime_error("Unknown model name: " + it.key());
                        auto index = size_t(found - names.begin());
                        if (std::string(section) == "objective") {
                            double value = it.value().get<double>();
                            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite objective");
                            candidate.c[index] = model.sense * value;
                        } else {
                            if (!it.value().is_object()) throw std::runtime_error("Bounds must be an object");
                            for (auto bound = it.value().begin(); bound != it.value().end(); ++bound) {
                                if (bound.key() != "lb" && bound.key() != "ub") throw std::runtime_error("Expected lb or ub");
                                bool lower = bound.key() == "lb";
                                double value = bound.value().is_null() ? (lower ? -inf : inf) : bound.value().get<double>();
                                if (!bound.value().is_null() && !std::isfinite(value)) throw std::runtime_error("Nonfinite bound");
                                auto &values = std::string(section) == "rows" ? (lower ? candidate.rl : candidate.ru) : (lower ? candidate.lb : candidate.ub);
                                values[index] = value;
                            }
                        }
                    }
                }
                candidate.validate();
                model = std::move(candidate);
                return json({{"status","UPDATED"},{"fingerprint",model.fingerprint()}});
            } else if (action == "solve") {
                Options options;
                options.device = request.value("device", "cpu");
                options.method = request.value("method", "auto");
                options.time_limit = request.value("time_limit", 60.);
                options.tol = request.value("tol", 1e-6);
                options.threads = request.value("threads", 1);
                options.iteration_limit = request.value("iterations", int64_t(100000));
                if (request.value("warm_start", true)) {
                    options.initial_x = previous.x;
                    options.initial_y = previous.y;
                    options.initial_basis = previous.basis;
                    options.basis_fingerprint = previous.basis_fingerprint;
                }
                interrupted = 0;
                auto result = solve(model, options);
                auto output = json::parse(result_json(model, result));
                output["session"] = {{"warm_start_requested",request.value("warm_start",true)}, {"model_retained",true}};
                if (result.accuracy.finite) previous = result;
                return output;
            } else throw std::runtime_error("Unknown session action");
    }
};
}
