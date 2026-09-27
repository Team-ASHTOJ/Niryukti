#include "internal.hpp"
#include <future>
#include <thread>
namespace vantage {
Result solve_portfolio(const Model &model, const Options &options) {
    auto start = Clock::now();
    auto cancellation = std::make_shared<std::atomic<bool>>(false);
    std::vector<std::future<Result>> workers;
    for (const auto &method : {"pdhg", "barrier", "simplex"}) {
        if (model.is_qp() && std::string(method) == "simplex")
            continue;
        Options o = options;
        o.method = method;
        o.cancellation = cancellation;
        if (o.method != "pdhg") {
            o.device = "cpu";
            o.cuda_graphs = false;
            o.matrix_precision = "fp64";
        }
        // Split the requested CPU thread budget instead of assigning it to every engine.
        o.threads = std::max(1, options.threads / (model.is_qp() ? 2 : 3));
        workers.push_back(std::async(std::launch::async, [&model, o] {
            try {
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
    best.message = "Concurrent engines; sibling cancellation uses local tokens. " + best.message;
    return best;
}
} // namespace vantage
