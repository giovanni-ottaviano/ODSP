#ifndef ODSP_DRIVER_PARALLEL_HPP
#define ODSP_DRIVER_PARALLEL_HPP

#include <algorithm>
#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace driver {

// Calls fn(i) for every i in [0, n) on at most `threads` threads (the calling
// thread does the work itself when threads <= 1). Indices are handed out one at
// a time, so uneven tasks balance out. The first exception thrown by fn stops
// handing out new indices and is rethrown once all threads have finished.
// fn must only write to data owned by its index.
inline void parallel_for(int n, int threads, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    threads = std::max(1, std::min(threads, n));
    if (threads == 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<int>   next{0};
    std::exception_ptr error;
    std::mutex         error_mutex;
    auto worker = [&]() {
        for (int i = next++; i < n; i = next++) {
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
                next = n;
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t)
        pool.emplace_back(worker);
    for (std::thread& t : pool)
        t.join();
    if (error)
        std::rethrow_exception(error);
}

} // namespace driver

#endif // ODSP_DRIVER_PARALLEL_HPP
