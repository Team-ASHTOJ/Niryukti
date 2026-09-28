#include "checkpoint.hpp"
#include "internal.hpp"
#include <future>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace vantage {
Result solve_portfolio(const Model &model, const Options &options) {
    auto start = Clock::now();
    auto cancellation = std::make_shared<std::atomic<bool>>(false);
    std::vector<std::future<Result>> workers;
    std::vector<std::string> methods = {"pdhg", "barrier"};
    if (!model.is_qp())
        methods.push_back("simplex");
    methods.resize(std::min<size_t>(methods.size(), std::max(1, options.threads)));
    CheckpointJson previous;
    if (!options.resume_path.empty()) {
        previous = read_engine_checkpoint(options.resume_path, "niryukti-portfolio-1",
                                          model.fingerprint(), options);
        if (previous.at("methods") != methods || previous.at("threads") != options.threads)
            throw std::runtime_error("Portfolio checkpoint engine/thread configuration mismatch");
    }
    std::vector<std::string> children;
    for (const auto &method : methods) {
        if (!options.checkpoint_path.empty())
            children.push_back(
                std::filesystem::absolute(options.checkpoint_path + "." + method + ".json")
                    .string());
    }
    if (!options.checkpoint_path.empty())
        write_engine_checkpoint(options.checkpoint_path,
                                {{"schema", "niryukti-portfolio-1"},
                                 {"fingerprint", model.fingerprint()},
                                 {"configuration", checkpoint_configuration(options)},
                                 {"iterations", 0},
                                 {"methods", methods},
                                 {"threads", options.threads},
                                 {"children", [&]() {
                                      std::vector<std::string> names;
                                      for (const auto &child : children)
                                          names.push_back(
                                              std::filesystem::path(child).filename().string());
                                      return names;
                                  }()}});
    for (size_t index = 0; index < methods.size(); ++index) {
        const auto &method = methods[index];
        Options o = options;
        o.method = method;
        o.cancellation = cancellation;
        o.checkpoint_path = children.empty() ? "" : children[index];
        if (!options.resume_path.empty()) {
            auto paths = previous.at("children").get<std::vector<std::string>>();
            if (paths.size() != methods.size())
                throw std::runtime_error("Portfolio checkpoint child dimensions");
            auto child_path =
                std::filesystem::path(options.resume_path).parent_path() / paths[index];
            if (!std::filesystem::is_regular_file(child_path))
                throw std::runtime_error("Portfolio checkpoint child state missing");
            o.resume_path = child_path.string();
        }
        if (o.method != "pdhg") {
            o.device = "cpu";
            o.cuda_graphs = false;
            o.matrix_precision = "fp64";
            o.gpu_presolve = false;
        }
        // Split the requested CPU thread budget instead of assigning it to every engine.
        o.threads = options.threads / int(methods.size()) +
                    (index < size_t(options.threads % int(methods.size())));
        workers.push_back(std::async(std::launch::async, [&model, o] {
            try {
#ifdef _OPENMP
                omp_set_num_threads(o.threads);
#endif
                return solve_continuous(model, o);
            } catch (const std::exception &error) {
                Result r;
                r.status = "NUMERICAL_ERROR";
                r.message = error.what();
                return r;
            }
        }));
    }
    Result best;
    bool winner = false;
    size_t remaining = workers.size();
    while (remaining) {
        bool progress = false;
        for (auto &worker : workers) {
            if (!worker.valid() ||
                worker.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
                continue;
            auto result = worker.get();
            --remaining;
            progress = true;
            bool optimal = result.status == "OPTIMAL" && result.x.size() == model.c.size() &&
                           result.y.size() == model.rl.size();
            if (optimal) {
                auto independent = verify(model, result.x, result.y);
                optimal = independent.finite && independent.kkt <= options.tol;
                if (optimal)
                    result.accuracy = independent;
            }
            if (optimal && !winner) {
                best = std::move(result);
                winner = true;
                cancellation->store(true, std::memory_order_relaxed);
            } else if (!winner && (result.accuracy.kkt < best.accuracy.kkt ||
                                   (best.status == "UNKNOWN" && result.status != "UNSUPPORTED")))
                best = std::move(result);
        }
        if (stop_requested(options))
            cancellation->store(true, std::memory_order_relaxed);
        if (!progress && remaining)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    best.seconds = elapsed(start);
    best.device_reason =
        "Concurrent portfolio selected " + best.method_selected + "; " + best.device_reason;
    best.message = std::to_string(methods.size()) +
                   " portfolio engines within requested CPU thread budget; sibling cancellation "
                   "uses local tokens. " +
                   best.message;
    return best;
}
} // namespace vantage
