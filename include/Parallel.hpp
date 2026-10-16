#ifndef ODSP_PARALLEL_HPP
#define ODSP_PARALLEL_HPP

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// parallel_for(n, threads, fn): calls fn(i) for every i in [0, n) on at most
// `threads` threads, and returns when all calls are done.
//
// Loops nest without oversubscribing the machine. The outermost loop starts a
// pool of `threads` workers (the calling thread is one of them). A loop
// started from inside a pool, e.g. the replicas of one sweep point, creates no
// threads: its `threads` argument is ignored, its indices join the same pool
// as a new job, and its caller works on them together with any idle workers.
// So the threads of a finished sweep point go on to help the points still
// running.
//
//   * Idle workers take an index from the oldest job that still has unstarted
//     ones: new sweep points are started first, then running ones are helped.
//   * A caller works on its own job first. While waiting for its last items
//     (running on other threads), it helps only jobs started after its own,
//     so it never picks up an outer item such as a whole new sweep point.
//   * Indices are handed out one at a time, so uneven items balance out.
//   * The first exception thrown by fn stops the handing out of that loop's
//     indices; it is rethrown to the loop's caller once its running items have
//     finished. Other loops are not affected.
//
// fn must only write to data owned by its index; then results do not depend
// on the number of threads. With threads <= 1 outside a pool, the calling
// thread does all the work itself (and loops nested in it are on their own).
namespace par {

namespace detail {

struct Job {
    int n = 0;
    const std::function<void(int)>* fn = nullptr;
    int next = 0;      // first unstarted index
    int running = 0;   // indices started but not finished
    std::exception_ptr error;

    bool available() const { return next < n; }
};

class Pool;
inline thread_local Pool* current_pool = nullptr;   // the pool this thread works for

class Pool {
public:
    // `threads` workers in all: threads - 1 new ones plus the caller of run().
    explicit Pool(int threads) {
        for (int t = 1; t < threads; ++t)
            _threads.emplace_back([this] { work(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stop = true;
        }
        _wake.notify_all();
        for (std::thread& t : _threads)
            t.join();
    }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    // Runs every index of `job`, with the calling thread taking part; returns
    // (or rethrows the job's first exception) when all of them have finished.
    void run(Job& job) {
        std::unique_lock<std::mutex> lock(_mutex);
        _jobs.push_back(&job);
        _wake.notify_all();
        for (;;) {
            Job* j = job.available() ? &job : nullptr;
            if (!j && job.running == 0)
                break;
            if (!j)
                j = first_available(std::find(_jobs.begin(), _jobs.end(), &job) + 1);   // later jobs only
            if (!j) {
                _wake.wait(lock);
                continue;
            }
            execute(*j, lock);
        }
        _jobs.erase(std::find(_jobs.begin(), _jobs.end(), &job));
        lock.unlock();
        if (job.error)
            std::rethrow_exception(job.error);
    }

private:
    Job* first_available(std::deque<Job*>::iterator from) {
        for (auto it = from; it != _jobs.end(); ++it)
            if ((*it)->available())
                return *it;
        return nullptr;
    }

    // Runs the next index of j; called and returns with the lock held.
    void execute(Job& j, std::unique_lock<std::mutex>& lock) {
        const int i = j.next++;
        ++j.running;
        lock.unlock();
        std::exception_ptr error;
        try {
            (*j.fn)(i);
        } catch (...) {
            error = std::current_exception();
        }
        lock.lock();
        if (error) {
            if (!j.error)
                j.error = error;
            j.next = j.n;   // stop handing out this job's indices
        }
        --j.running;
        _wake.notify_all();
    }

    void work() {
        current_pool = this;
        std::unique_lock<std::mutex> lock(_mutex);
        for (;;) {
            if (Job* j = first_available(_jobs.begin()))
                execute(*j, lock);
            else if (_stop)
                return;
            else
                _wake.wait(lock);
        }
    }

    std::mutex _mutex;               // guards everything below and every Job's counters
    std::condition_variable _wake;   // a job was added, an index finished, or stop
    std::deque<Job*> _jobs;          // in the order they were started
    bool _stop = false;
    std::vector<std::thread> _threads;
};

} // namespace detail

inline void parallel_for(int n, int threads, const std::function<void(int)>& fn) {
    if (n <= 0)
        return;
    detail::Job job;
    job.n = n;
    job.fn = &fn;
    if (detail::current_pool) {   // nested: join the enclosing pool
        detail::current_pool->run(job);
        return;
    }
    if (threads <= 1) {
        for (int i = 0; i < n; ++i)
            fn(i);
        return;
    }
    detail::Pool pool(threads);   // all `threads`, not min(threads, n): nested loops may use them
    detail::current_pool = &pool;
    try {
        pool.run(job);
    } catch (...) {
        detail::current_pool = nullptr;
        throw;
    }
    detail::current_pool = nullptr;
}

} // namespace par

#endif // ODSP_PARALLEL_HPP
