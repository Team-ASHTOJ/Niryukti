#pragma once
#include "internal.hpp"
#include "json.hpp"
#include <filesystem>
#include <fstream>
namespace vantage {
using CheckpointJson = nlohmann::json;
inline CheckpointJson checkpoint_configuration(const Options &o) {
    return {{"presolve", o.presolve},
            {"scaling_passes", o.scaling_passes},
            {"scaling", o.scaling},
            {"tolerance", o.tol},
            {"device", o.device}};
}
inline CheckpointJson read_engine_checkpoint(const std::string &path, const std::string &schema,
                                             const std::string &fingerprint, const Options &o) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open solver checkpoint");
    CheckpointJson saved;
    file >> saved;
    if (saved.at("schema") != schema || saved.at("fingerprint") != fingerprint ||
        saved.at("configuration") != checkpoint_configuration(o))
        throw std::runtime_error("Solver checkpoint model/configuration mismatch");
    if (!saved.at("iterations").is_number_integer() || saved.at("iterations").get<int64_t>() < 0)
        throw std::runtime_error("Invalid checkpoint iteration counter");
    return saved;
}
inline void write_engine_checkpoint(const std::string &path, const CheckpointJson &saved) {
    if (path.empty())
        return;
    auto temporary = path + ".tmp";
    std::ofstream file(temporary);
    if (!file)
        throw std::runtime_error("Cannot write solver checkpoint");
    file << saved.dump();
    file.flush();
    if (!file)
        throw std::runtime_error("Solver checkpoint write failed");
    file.close();
    std::filesystem::rename(temporary, path);
}
} // namespace vantage
