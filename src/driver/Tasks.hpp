#ifndef ODSP_DRIVER_TASKS_HPP
#define ODSP_DRIVER_TASKS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "CsvWriter.hpp"
#include "Ensemble.hpp"
#include "Setup.hpp"
#include "Options.hpp"
#include "Parallel.hpp"

// The three tasks of the driver (run, ensemble, stationary) and parameter
// sweeps over the last two. Each task prints a readable summary and, given a
// CSV prefix, writes its data; ensemble and stationary also return a summary
// row (name, value) that a sweep collects into one table.

namespace driver {

using Row = std::vector<std::pair<std::string, double>>;

struct TaskParams {
    std::string task = "run";
    long long   steps = 1000;
    long long   every = 1;
    bool        stop = true;
    std::vector<std::string> observables;
    int         replicas = 100;
    bool        sweeps_unit = true;
    std::string observable;
    long long   burn_in = 1000;
    int         samples = 1000;
    int         interval = 1;
    int         blocks = 20;
    int         bins = 50;
    int         chains = 1;                 // stationary: independent chains (--replicas)
    double      threshold = std::nan("");   // ensemble: largest-share threshold (NaN = none)
    std::vector<long long> times;           // recording times (run; ensemble time series if non-empty)
    bool        correlation = false;        // write G(r) curves (--correlation)
};

// Observables measured against a snapshot at waiting time t_w ("_tw<t>").
inline bool is_two_time(const std::string& name) { return tw_of(name) >= 0; }

// Expands "autocorrelation" / "overlap" into one observable per --tw value and
// adds the waiting times to the recording grid (so the snapshots are taken
// exactly at t_w). Only reads --tw when such an observable was asked for.
inline void expand_two_time(Options& o, TaskParams& t) {
    bool wanted = false;
    for (const auto& n : t.observables) wanted |= n == "autocorrelation" || n == "overlap";
    if (!wanted) return;
    if (!o.has("tw")) throw OptionError("observables autocorrelation/overlap need waiting times: --tw t1,t2,...");
    std::vector<long long> tws;
    for (const std::string& v : o.word_list("tw", "")) {
        try {
            std::size_t used = 0;
            const long long tw = std::stoll(v, &used);
            if (used != v.size() || tw < 0 || tw > t.steps) throw std::out_of_range(v);
            tws.push_back(tw);
        } catch (const std::exception&) {
            throw OptionError("option --tw: expected waiting times in 0.." + std::to_string(t.steps) + ", got '" + v + "'");
        }
    }
    std::vector<std::string> expanded;
    for (const auto& n : t.observables) {
        if (n != "autocorrelation" && n != "overlap") { expanded.push_back(n); continue; }
        for (long long tw : tws) expanded.push_back(n + "_tw" + std::to_string(tw));
    }
    t.observables = expanded;
    if (!t.times.empty()) {
        for (long long tw : tws) t.times.push_back(tw);   // (a range insert trips a GCC 13 false-positive warning)
        std::sort(t.times.begin(), t.times.end());
        t.times.erase(std::unique(t.times.begin(), t.times.end()), t.times.end());
    }
}

// Times at which to record, from 0 to steps (both included), strictly
// increasing: every `every` steps, or -- when per_decade > 0 -- log-spaced with
// that many points per decade (10^(i/per_decade), rounded, duplicates dropped).
inline std::vector<long long> recording_times(long long steps, long long every, int per_decade) {
    std::vector<long long> times = {0};
    if (per_decade > 0) {
        for (int i = 0;; ++i) {
            const long long t = std::llround(std::pow(10.0, static_cast<double>(i) / per_decade));
            if (t >= steps) break;
            if (t > times.back()) times.push_back(t);
        }
    } else {
        const long long e = std::max<long long>(every, 1);
        for (long long t = e; t < steps; t += e) times.push_back(t);
    }
    if (steps > times.back()) times.push_back(steps);
    return times;
}

// --log-times [k]: k points per decade (bare flag = 10); 0 if not given.
inline int read_log_times(Options& o) {
    if (!o.has("log-times")) return 0;
    const std::string v = o.str("log-times");
    if (v == "true") return 10;
    try {
        std::size_t used = 0;
        const int k = std::stoi(v, &used);
        if (used == v.size() && k >= 1) return k;
    } catch (const std::exception&) {}
    throw OptionError("option --log-times: expected a number of points per decade >= 1, got '" + v + "'");
}

inline TaskParams read_task(Options& o, const std::string& task, const ModelSpec& m, const InitSpec& init) {
    TaskParams t;
    t.task = task;
    auto positive = [&o](const std::string& key, long long fallback) {
        const long long v = o.integer(key, fallback);
        if (v < 1) throw OptionError("option --" + key + " must be >= 1");
        return v;
    };
    // Recording grid: --every n or --log-times [k], not both.
    auto read_grid = [&](long long default_every) -> std::vector<long long> {
        if (o.has("every") && o.has("log-times")) throw OptionError("use either --every or --log-times, not both");
        const int per_decade = read_log_times(o);
        if (per_decade > 0) return recording_times(t.steps, 0, per_decade);
        if (default_every == 0 && !o.has("every")) return {};
        t.every = positive("every", default_every > 0 ? default_every : 1);
        return recording_times(t.steps, t.every, 0);
    };
    if (task == "run") {
        t.steps = positive("steps", 1000);
        t.times = read_grid(1);
        t.stop = o.flag("stop", true);
        const auto d = default_observables(m, init, task);
        std::string joined;
        for (const auto& n : d) joined += (joined.empty() ? "" : ",") + n;
        t.observables = o.word_list("observables", joined);
        expand_two_time(o, t);
        t.correlation = o.flag("correlation", false);
    } else if (task == "ensemble") {
        t.replicas = static_cast<int>(positive("replicas", 100));
        t.sweeps_unit = o.choice("time-unit", "sweeps", {"sweeps", "updates"}) == "sweeps";
        t.steps = positive("steps", t.sweeps_unit ? 10000 : 100000000LL);
        t.times = read_grid(0);   // empty: no time series
        if (o.has("threshold")) {
            if (is_continuous(m.name)) throw OptionError("option --threshold needs discrete opinions (model " + m.name + ")");
            t.threshold = o.num("threshold");
            if (!(t.threshold > 0.0 && t.threshold <= 1.0)) throw OptionError("option --threshold must be in (0, 1]");
        }
        const auto d = default_observables(m, init, task);
        std::string joined;
        for (const auto& n : d) joined += (joined.empty() ? "" : ",") + n;
        t.observables = o.word_list("observables", joined);
        expand_two_time(o, t);
        t.correlation = o.flag("correlation", false);
        for (const auto& n : t.observables)
            if (is_two_time(n) && t.times.empty())
                throw OptionError("observable " + n + " needs an ensemble time series: add --every or --log-times");
        if (t.correlation && t.times.empty())
            throw OptionError("--correlation in an ensemble needs a time series: add --every or --log-times");
    } else {   // stationary
        t.observable = o.str("observable", default_observables(m, init, task).front());
        t.burn_in = o.integer("burn-in", 1000);
        if (t.burn_in < 0) throw OptionError("option --burn-in must be >= 0");
        t.samples = static_cast<int>(positive("samples", 1000));
        t.interval = static_cast<int>(positive("interval", 1));
        t.blocks = static_cast<int>(positive("blocks", 20));
        t.bins = static_cast<int>(positive("bins", 50));
        t.chains = static_cast<int>(positive("replicas", 1));
        t.correlation = o.flag("correlation", false);
    }
    return t;
}

inline std::string path_for(const std::string& prefix, const std::string& name) {
    return prefix.empty() ? std::string() : prefix + "_" + name + ".csv";
}

inline std::string fmt(double v, int precision = 6) {
    std::ostringstream os;
    os << std::setprecision(precision) << v;
    return os.str();
}

// ----------------------------------------------------------------------- run

template <typename T>
void task_run(const TaskParams& t, const Setup<T>& s, unsigned seed, const std::string& prefix) {
    std::mt19937 rng(seed);
    auto model = s.factory(rng);
    std::vector<Observable<T>> obs;
    for (const auto& name : t.observables)
        obs.push_back(make_observable<T>(name, s.model, /*per_interval=*/true, s.context));

    std::vector<std::string> header = {"sweep"};
    header.insert(header.end(), t.observables.begin(), t.observables.end());
    CsvWriter csv(path_for(prefix, "timeseries"), header);
    // G(r) at every recording time, long format: sweep, r, G.
    CsvWriter curves(t.correlation ? path_for(prefix, "correlation") : std::string(), {"sweep", "r", "G"});

    std::vector<int> widths;   // each column at least as wide as its name
    for (const auto& n : t.observables) widths.push_back(static_cast<int>(std::max<std::size_t>(14, n.size() + 2)));
    std::cout << std::setw(10) << "sweep";
    for (std::size_t k = 0; k < t.observables.size(); ++k) std::cout << std::setw(widths[k]) << t.observables[k];
    std::cout << '\n';
    long long next_print = 1;
    auto record = [&](long long sweep, bool force_print) {
        std::vector<double> values;
        for (const auto& f : obs) values.push_back(f(*model));
        csv.row_values(std::to_string(sweep), values);
        if (t.correlation) {
            const auto G = corr::spatial_correlation(static_cast<const RegularLattice<T>&>(model->get_graph()));
            for (std::size_t r = 0; r < G.size(); ++r) curves.row(sweep, r, G[r]);
        }
        if (sweep == 0 || sweep >= next_print || force_print) {   // print on a log scale
            std::cout << std::setw(10) << sweep;
            for (std::size_t k = 0; k < values.size(); ++k) std::cout << std::setw(widths[k]) << fmt(values[k]);
            std::cout << '\n';
            while (next_print <= sweep) next_print *= 2;
        }
    };

    record(0, false);
    long long sweep = 0;
    std::size_t next = 1;   // index of the next recording time
    bool absorbed = false;
    while (sweep < t.steps) {
        model->advance(1, true, false);
        ++sweep;
        absorbed = model->is_absorbing_state();
        const bool last = sweep == t.steps || (t.stop && absorbed);
        const bool due = next < t.times.size() && t.times[next] == sweep;
        if (due) ++next;
        if (due || last) record(sweep, last);
        if (t.stop && absorbed) break;
    }
    std::cout << "\nStopped after " << sweep << " sweeps"
              << (absorbed ? (model->is_consensus_reached() ? " in consensus" : ", converged (no further change)")
                           : (model->is_consensus_reached() ? " (in consensus at this moment)" : ""))
              << ".\n";
    if (!prefix.empty()) std::cout << "Time series written to " << path_for(prefix, "timeseries") << '\n';
}

// ------------------------------------------------------------------ ensemble

template <typename T>
Row task_ensemble(const TaskParams& t, const Setup<T>& s, unsigned seed, int threads, const std::string& prefix,
                  bool verbose) {
    ObservableContext ctx = s.context;
    ctx.sweeps = t.sweeps_unit;
    // Final values: every observable except the two-time ones, which follow a
    // trajectory (their last value is in the time series).
    std::vector<std::string> final_names;
    for (const auto& name : t.observables)
        if (!is_two_time(name)) final_names.push_back(name);
    std::vector<typename Ensemble<T>::FinalObservable> obs;
    for (const auto& name : final_names) obs.push_back(make_observable<T>(name, s.model, false, ctx));
    // Hidden last observable: did the replica end absorbed (consensus, or
    // convergence for continuous models) rather than at the step cap?
    obs.push_back([](const Model<T>& m) { return m.is_absorbing_state() ? 1.0 : 0.0; });

    typename Ensemble<T>::Hooks hooks;
    const std::size_t R = static_cast<std::size_t>(t.replicas);
    const bool sweeps = t.sweeps_unit;

    // Time at which the largest opinion share first reaches the threshold (-1: never).
    std::vector<long long> threshold_times(R, -1);
    const bool use_threshold = !std::isnan(t.threshold);
    const double x = t.threshold;

    // Time series: each replica records every observable at the grid times into
    // its own slots; after it stops, its final state fills the remaining times
    // (an absorbed state never changes again). Observables are created per
    // replica, so per-interval ones (activity) keep separate state.
    const std::vector<long long>& grid = t.times;
    const std::size_t K = grid.size(), O = t.observables.size();
    const bool series = K > 0;
    std::vector<double> series_values(series ? R * K * O : 0);
    std::vector<std::size_t> recorded(R, 0);
    std::vector<std::vector<Observable<T>>> replica_obs(R);
    auto slot = [&series_values, K, O](std::size_t r, std::size_t k, std::size_t o) -> double& {
        return series_values[(r * K + k) * O + o];
    };
    // G(r) per replica and recording time, when --correlation is given.
    std::vector<std::vector<double>> curves(t.correlation ? R * K : 0);
    auto curve_of = [](const Model<T>& m) {
        return corr::spatial_correlation(static_cast<const RegularLattice<T>&>(m.get_graph()));
    };

    if (use_threshold || series) {
        hooks.step = [&, x, sweeps, use_threshold, series](int replica) {
            const std::size_t r = static_cast<std::size_t>(replica);
            if (series)
                for (const auto& name : t.observables) replica_obs[r].push_back(make_observable<T>(name, s.model, true, ctx));
            return typename Model<T>::Observer([&, r, x, sweeps, use_threshold, series](const Model<T>& m) {
                const long long now = sweeps ? m.get_mc_steps() : m.get_updates();
                if constexpr (std::is_integral<T>::value)
                    if (use_threshold && threshold_times[r] < 0 && m.largest_opinion_fraction() >= x - 1e-12)
                        threshold_times[r] = now;
                if (series && recorded[r] < K && grid[recorded[r]] == now) {
                    for (std::size_t o = 0; o < O; ++o) slot(r, recorded[r], o) = replica_obs[r][o](m);
                    if (t.correlation) curves[r * K + recorded[r]] = curve_of(m);
                    ++recorded[r];
                }
            });
        };
    }
    // Continuous opinions: keep each replica's final clusters.
    std::vector<std::vector<OpinionCluster>> clusters(R);
    const bool keep_clusters = is_continuous(s.model.name) && !prefix.empty();
    if (series || keep_clusters)
        hooks.finish = [&, series, keep_clusters](int replica, const Model<T>& m) {
            const std::size_t r = static_cast<std::size_t>(replica);
            for (; series && recorded[r] < K; ++recorded[r]) {
                for (std::size_t o = 0; o < O; ++o) slot(r, recorded[r], o) = replica_obs[r][o](m);
                if (t.correlation) curves[r * K + recorded[r]] = curve_of(m);
            }
            if (keep_clusters) {
                const auto& xs = m.get_graph().states();
                clusters[r] = opinion_clusters(std::vector<double>(xs.begin(), xs.end()), 1e-3);
            }
        };

    Ensemble<T> ens(s.factory, seed);
    const auto r = ens.run(t.replicas, t.steps, t.sweeps_unit, threads, obs, hooks);
    const std::vector<double>& absorbed_flags = r.final_observables.back();
    obs.pop_back();
    int absorbed = 0;
    for (double a : absorbed_flags) absorbed += a > 0.5 ? 1 : 0;
    const std::string unit = t.sweeps_unit ? "sweeps" : "updates";

    Row row = {{"replicas", r.replicas},
               {"consensus_fraction", static_cast<double>(r.consensus_count) / r.replicas},
               {"absorbed_fraction", static_cast<double>(absorbed) / r.replicas},
               {"mean_consensus_time", r.mean_consensus_time},
               {"sd_consensus_time", r.stddev_consensus_time},
               {"median_consensus_time", r.median_consensus_time},
               {"mean_consensus_time_lower_bound", r.mean_consensus_time_lower_bound}};
    if (!s.init.multi && !is_continuous(s.model.name) && !r.first_passage) row.push_back({"exit_prob_up", r.exit_prob_up});
    int reached_threshold = 0;
    double sum_threshold = 0.0;
    for (long long v : threshold_times) if (v >= 0) { ++reached_threshold; sum_threshold += static_cast<double>(v); }
    if (use_threshold) {
        row.push_back({"threshold_fraction", static_cast<double>(reached_threshold) / r.replicas});
        row.push_back({"mean_threshold_time", reached_threshold ? sum_threshold / reached_threshold : std::nan("")});
    }

    std::vector<double> means, sds;
    for (std::size_t k = 0; k < obs.size(); ++k) {
        double sum = 0.0, sum2 = 0.0;
        for (double v : r.final_observables[k]) { sum += v; sum2 += v * v; }
        means.push_back(sum / r.replicas);
        sds.push_back(std::sqrt(std::max(0.0, sum2 / r.replicas - means[k] * means[k])));
        row.push_back({"mean_" + final_names[k], means[k]});
        row.push_back({"sd_" + final_names[k], sds[k]});
    }

    if (!prefix.empty()) {
        std::vector<std::string> header = {"replica", "reached_consensus", "absorbed", "stop_time"};
        if (use_threshold) header.push_back("threshold_time");
        header.insert(header.end(), final_names.begin(), final_names.end());
        CsvWriter rep(path_for(prefix, "replicas"), header);
        for (int i = 0; i < r.replicas; ++i) {
            std::vector<double> values = {static_cast<double>(r.reached[i]), absorbed_flags[i], static_cast<double>(r.stop_times[i])};
            if (use_threshold) values.push_back(static_cast<double>(threshold_times[static_cast<std::size_t>(i)]));
            for (std::size_t k = 0; k < obs.size(); ++k) values.push_back(r.final_observables[k][i]);
            rep.row_values(std::to_string(i), values);
        }
        if (is_continuous(s.model.name)) {
            CsvWriter cl(path_for(prefix, "clusters"), {"replica", "position", "size", "min", "max"});
            for (std::size_t i = 0; i < clusters.size(); ++i)
                for (const OpinionCluster& c : clusters[i]) cl.row(i, c.position, c.size, c.min, c.max);
        }
        if (!r.fixation_counts.empty()) {
            CsvWriter fix(path_for(prefix, "fixation"), {"opinion", "replicas", "probability"});
            for (const auto& e : r.fixation_counts)
                fix.row(e.first, e.second, static_cast<double>(e.second) / r.consensus_count);
        }
    }

    // ---- time series: averages over replicas at each recording time ----
    if (series) {
        // Per replica: when consensus was (first) reached, -1 if never.
        std::vector<long long> consensus_at(R, -1);
        for (std::size_t i = 0, j = 0; i < R; ++i)
            if (r.reached[i]) consensus_at[i] = r.consensus_times[j++];
        std::vector<std::string> header = {"time", "running_fraction", "survival"};
        for (const auto& n : t.observables) {
            header.push_back(n + "_mean");
            header.push_back(n + "_err");
            header.push_back(n + "_mean_running");
        }
        CsvWriter csv(path_for(prefix, "timeseries"), header);
        std::vector<std::vector<double>> printed;   // per time: means of the observables
        for (std::size_t k = 0; k < K; ++k) {
            std::size_t running = 0, surviving = 0;
            std::vector<char> is_running(R);
            for (std::size_t i = 0; i < R; ++i) {
                is_running[i] = !(absorbed_flags[i] > 0.5 && r.stop_times[i] <= grid[k]);
                running += is_running[i];
                surviving += !(consensus_at[i] >= 0 && consensus_at[i] <= grid[k]);
            }
            std::vector<double> values = {static_cast<double>(running) / R, static_cast<double>(surviving) / R};
            std::vector<double> means;
            for (std::size_t o = 0; o < O; ++o) {
                double sum = 0.0, sum2 = 0.0, sum_run = 0.0;
                for (std::size_t i = 0; i < R; ++i) {
                    const double v = slot(i, k, o);
                    sum += v; sum2 += v * v;
                    if (is_running[i]) sum_run += v;
                }
                const double mean = sum / R;
                const double sd = std::sqrt(std::max(0.0, sum2 / R - mean * mean));
                values.push_back(mean);
                values.push_back(sd / std::sqrt(static_cast<double>(R)));
                values.push_back(running ? sum_run / running : std::nan(""));
                means.push_back(mean);
            }
            csv.row_values(std::to_string(grid[k]), values);
            printed.push_back(means);
        }
        if (verbose) {
            std::cout << "Time series (mean over all " << R << " runs; absorbed runs keep their final value):\n"
                      << std::setw(12) << unit << std::setw(10) << "survival";
            for (const auto& n : t.observables) std::cout << std::setw(static_cast<int>(std::max<std::size_t>(14, n.size() + 2))) << n;
            std::cout << '\n';
            const std::size_t rows = std::min<std::size_t>(K, 25);
            std::size_t last = K;   // print at most 25 rows, evenly spread over the grid
            for (std::size_t q = 0; q < rows; ++q) {
                const std::size_t k = rows == 1 ? 0 : q * (K - 1) / (rows - 1);
                if (k == last) continue;
                last = k;
                std::size_t surviving = 0;
                for (std::size_t i = 0; i < R; ++i) surviving += !(consensus_at[i] >= 0 && consensus_at[i] <= grid[k]);
                std::cout << std::setw(12) << grid[k] << std::setw(10) << fmt(static_cast<double>(surviving) / R, 4);
                for (std::size_t o = 0; o < O; ++o)
                    std::cout << std::setw(static_cast<int>(std::max<std::size_t>(14, t.observables[o].size() + 2)))
                              << fmt(printed[k][o]);
                std::cout << '\n';
            }
            if (!prefix.empty()) std::cout << "Full time series (with errors and survivor means) in " << path_for(prefix, "timeseries") << '\n';
            std::cout << '\n';
        }
    }

    // ---- G(r) curves: mean and standard error over replicas at each time ----
    if (t.correlation && !prefix.empty()) {
        CsvWriter out(path_for(prefix, "correlation"), {"time", "r", "G_mean", "G_err"});
        for (std::size_t k = 0; k < K; ++k) {
            const std::size_t len = curves[k].size();   // replica 0's curve fixes the length
            for (std::size_t d = 0; d < len; ++d) {
                double sum = 0.0, sum2 = 0.0;
                for (std::size_t i = 0; i < R; ++i) {
                    const double g = curves[i * K + k][d];
                    sum += g; sum2 += g * g;
                }
                const double mean = sum / R, sd = std::sqrt(std::max(0.0, sum2 / R - mean * mean));
                out.row(grid[k], d, mean, sd / std::sqrt(static_cast<double>(R)));
            }
        }
    }

    if (verbose) {
        std::cout << "Replicas: " << r.replicas << ", each run for " << (r.first_passage ? "" : "at most ")
                  << t.steps << " " << unit << ".\n";
        if (r.first_passage)
            std::cout << "This model can leave consensus, so consensus times are FIRST-PASSAGE times and there is\n"
                         "no fixation; final values are measured at the cap.\n";
        std::cout << (r.first_passage ? "Consensus reached at least once: " : "Consensus reached: ") << r.consensus_count
                  << " (" << fmt(100.0 * r.consensus_count / r.replicas, 4) << "%)\n";
        if (!r.first_passage && absorbed != r.consensus_count)
            std::cout << "Stopped in an absorbing state (no further change possible): " << absorbed << " ("
                      << fmt(100.0 * absorbed / r.replicas, 4) << "%)"
                      << (is_continuous(s.model.name) ? " -- converged, possibly into several clusters" : "") << '\n';
        if (r.consensus_count > 0) {
            std::cout << (r.first_passage ? "First-passage time (" : "Consensus time (") << unit << "): mean " << fmt(r.mean_consensus_time) << " +- "
                      << fmt(r.stddev_consensus_time / std::sqrt(static_cast<double>(r.consensus_count)), 3)
                      << ", sd " << fmt(r.stddev_consensus_time) << ", median "
                      << (std::isnan(r.median_consensus_time) ? std::string("n/a") : fmt(r.median_consensus_time)) << '\n';
            if (r.consensus_count < r.replicas)
                std::cout << "  (" << r.replicas - r.consensus_count << " runs did not reach consensus: the mean is biased low;"
                          << " true mean >= " << fmt(r.mean_consensus_time_lower_bound) << ")\n";
            if (!is_continuous(s.model.name) && !r.first_passage) {
                std::vector<std::pair<T, int>> fix = r.fixation_counts;
                std::sort(fix.begin(), fix.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
                std::cout << "Winning opinion:";
                for (std::size_t k = 0; k < fix.size() && k < 10; ++k)
                    std::cout << "  " << fix[k].first << ": " << fmt(static_cast<double>(fix[k].second) / r.consensus_count, 4);
                if (fix.size() > 10) std::cout << "  ... (" << fix.size() << " opinions won at least once)";
                std::cout << '\n';
            }
        }
        if (use_threshold)
            std::cout << "Largest opinion share reached " << fmt(t.threshold, 4) << ": " << reached_threshold << " replicas ("
                      << fmt(100.0 * reached_threshold / r.replicas, 4) << "%)"
                      << (reached_threshold ? ", mean time " + fmt(sum_threshold / reached_threshold) + " " + unit : std::string())
                      << '\n';
        if (!obs.empty()) {
            std::cout << "Final values (mean +- standard error, sd):\n";
            for (std::size_t k = 0; k < obs.size(); ++k)
                std::cout << "  " << std::left << std::setw(static_cast<int>(std::max<std::size_t>(16, final_names[k].size() + 2)))
                          << final_names[k] << std::right << fmt(means[k]) << " +- "
                          << fmt(sds[k] / std::sqrt(static_cast<double>(r.replicas)), 3) << "   (sd " << fmt(sds[k], 4) << ")\n";
        }
        if (!prefix.empty())
            std::cout << "Per-replica data written to " << path_for(prefix, "replicas")
                      << (is_continuous(s.model.name) ? ", final clusters to " + path_for(prefix, "clusters") : std::string())
                      << (r.fixation_counts.empty() ? "" : ", fixation counts to " + path_for(prefix, "fixation")) << '\n';
    }
    return row;
}

// ---------------------------------------------------------------- stationary

template <typename T>
Row task_stationary(const TaskParams& t, const Setup<T>& s, unsigned seed, const std::string& prefix, bool verbose,
                    int threads = 1) {
    // Independent chains, run in parallel. Chain 0 uses `seed` itself, so a
    // single chain reproduces the plain single-chain run exactly.
    const int R = t.chains;
    std::vector<unsigned> seeds(static_cast<std::size_t>(R), seed);
    std::mt19937 seeder(seed);
    for (int c = 1; c < R; ++c) seeds[static_cast<std::size_t>(c)] = seeder();

    const bool thermal = s.model.name == "ising" || s.model.name == "potts";   // energy -> specific heat
    // Second-moment correlation length: periodic hypercubic lattice, discrete
    // opinions with a fixed set of at least two labels.
    const ObservableContext& ctx = s.context;
    const bool want_xi = std::is_integral<T>::value && ctx.periodic_cubic && ctx.alphabet.size() >= 2;
    struct Chain {
        typename Model<T>::StationarySample st;
        std::vector<double> energy, s0, s_min;    // per sample
        std::vector<std::vector<double>> curves;  // G(r) per sample (--correlation)
    };
    std::vector<Chain> chains(static_cast<std::size_t>(R));
    double N = 0.0;
    bool absorbing = false;
    parallel_for(R, std::min(std::max(threads, 1), R), [&](int c) {
        Chain& ch = chains[static_cast<std::size_t>(c)];
        std::mt19937 rng(seeds[static_cast<std::size_t>(c)]);
        auto model = s.factory(rng);
        if (c == 0) {
            N = static_cast<double>(model->get_graph().size());
            absorbing = model->consensus_is_absorbing();
        }
        const Observable<T> base = make_observable<T>(t.observable, s.model, /*per_interval=*/true, ctx);
        Observable<T> energy_of;
        if constexpr (std::is_integral<T>::value)
            if (thermal) energy_of = make_observable<T>("energy", s.model);
        const std::vector<T> labels(ctx.alphabet.begin(), ctx.alphabet.end());
        // Side series sampled with the main observable.
        const Observable<T> obs = [&, labels](const Model<T>& m) {
            if (thermal) ch.energy.push_back(energy_of(m));
            if (want_xi || t.correlation) {
                const auto& lattice = static_cast<const RegularLattice<T>&>(m.get_graph());
                if (want_xi) {
                    const corr::StructureFactor sf = corr::structure_factor(lattice, labels);
                    ch.s0.push_back(sf.s0);
                    ch.s_min.push_back(sf.s_min);
                }
                if (t.correlation) ch.curves.push_back(corr::spatial_correlation(lattice));
            }
            return base(m);
        };
        ch.st = model->sample_stationary(obs, t.burn_in, t.samples, t.interval, true, t.blocks);
    });

    // Pooled samples, chain after chain (equal lengths), so a jackknife with
    // one block per chain is leave-one-chain-out.
    std::vector<double> pooled, energy, s0, s_min;
    for (const Chain& ch : chains) {
        pooled.insert(pooled.end(), ch.st.m.begin(), ch.st.m.end());
        energy.insert(energy.end(), ch.energy.begin(), ch.energy.end());
        s0.insert(s0.end(), ch.s0.begin(), ch.s0.end());
        s_min.insert(s_min.end(), ch.s_min.begin(), ch.s_min.end());
    }
    using Estimator = std::function<double(const stats::Moments&)>;
    const double T2 = s.model.temperature * s.model.temperature;
    const std::vector<std::pair<std::string, Estimator>> estimators = {
        {"mean", [](const stats::Moments& m) { return m.m1; }},
        {"abs_mean", [](const stats::Moments& m) { return m.abs_m1; }},
        {"susceptibility", [N](const stats::Moments& m) { return N * (m.m2 - m.m1 * m.m1); }},
        {"abs_susceptibility", [N](const stats::Moments& m) { return N * (m.m2 - m.abs_m1 * m.abs_m1); }},
        {"binder", [](const stats::Moments& m) { return m.m2 > 0.0 ? 1.0 - m.m4 / (3.0 * m.m2 * m.m2) : 0.0; }},
    };
    const Estimator e_mean_of = [](const stats::Moments& m) { return m.m1; };
    const Estimator heat_of = [N, T2](const stats::Moments& m) { return N * (m.m2 - m.m1 * m.m1) / T2; };

    // Error of an estimator: one chain -> jackknife within it; several chains
    // -> leave-one-chain-out. "within": the chains' own jackknife errors,
    // combined as for an average of R independent estimates.
    auto between = [&](const std::vector<double>& series, const Estimator& f) {
        return R == 1 ? stats::jackknife_error(series, t.blocks, f) : stats::jackknife_error(series, R, f);
    };
    auto within = [&](bool use_energy, const Estimator& f) {
        double sum2 = 0.0;
        for (const Chain& ch : chains) {
            const double e = stats::jackknife_error(use_energy ? ch.energy : ch.st.m, t.blocks, f);
            sum2 += e * e;
        }
        return std::sqrt(sum2) / R;
    };

    const stats::Moments mo = stats::moments(pooled);
    double tau = 0.0, tau_abs = 0.0;
    for (const Chain& ch : chains) { tau += ch.st.tau_m / R; tau_abs += ch.st.tau_abs_m / R; }

    Row row;
    std::vector<double> estimate, error, error_within;
    for (const auto& [name, f] : estimators) {
        // With one chain take the model's own numbers: identical to the plain run.
        const auto& st = chains[0].st;
        const double v = R > 1 ? f(mo) : name == "mean" ? st.mean : name == "abs_mean" ? st.abs_mean
                       : name == "susceptibility" ? st.susceptibility : name == "abs_susceptibility" ? st.abs_susceptibility
                       : st.binder;
        const double e = R > 1 ? between(pooled, f) : name == "mean" ? st.mean_err : name == "abs_mean" ? st.abs_mean_err
                       : name == "susceptibility" ? st.susceptibility_err
                       : name == "abs_susceptibility" ? st.abs_susceptibility_err : st.binder_err;
        const double w = R > 1 ? within(false, f) : e;
        estimate.push_back(v); error.push_back(e); error_within.push_back(w);
        row.push_back({name, v});
        row.push_back({name + "_err", e});
        if (name == "abs_mean") { row.push_back({"second_moment", mo.m2}); row.push_back({"fourth_moment", mo.m4}); }
    }
    row.push_back({"tau", tau});
    row.push_back({"tau_abs", tau_abs});
    row.push_back({"chains", static_cast<double>(R)});
    for (std::size_t k = 0; k < estimators.size(); ++k) row.push_back({estimators[k].first + "_err_within", error_within[k]});

    // Specific heat per site C = N var(e) / T^2, e = energy per site.
    double e_mean = 0.0, e_err = 0.0, heat = 0.0, heat_err = 0.0, heat_within = 0.0;
    if (thermal && !energy.empty()) {
        const stats::Moments em = stats::moments(energy);
        e_mean = em.m1;
        e_err = between(energy, e_mean_of);
        heat = heat_of(em);
        heat_err = between(energy, heat_of);
        heat_within = R > 1 ? within(true, heat_of) : heat_err;
        row.insert(row.end(), {{"energy", e_mean}, {"energy_err", e_err}, {"specific_heat", heat},
                               {"specific_heat_err", heat_err}, {"specific_heat_err_within", heat_within}});
    }

    // Second-moment correlation length from the averaged structure factors
    // (a ratio of averages, so its error needs the two-series jackknife).
    double xi = std::nan(""), xi_err = std::nan("");
    if (want_xi && !s0.empty()) {
        const int L = ctx.side;
        auto xi_of = [L](double a, double b) { return corr::second_moment_length(a, b, L); };
        const stats::Moments m0 = stats::moments(s0), m1 = stats::moments(s_min);
        xi = xi_of(m0.m1, m1.m1);
        xi_err = R == 1 ? stats::jackknife_error2(s0, s_min, t.blocks, xi_of) : stats::jackknife_error2(s0, s_min, R, xi_of);
        row.insert(row.end(), {{"xi", xi}, {"xi_err", xi_err}, {"xi_over_L", xi / L}, {"xi_over_L_err", xi_err / L}});
    }

    // G(r) averaged over all samples; error from block means (one block per
    // chain with several chains, else --blocks blocks of the single chain).
    if (t.correlation && !prefix.empty() && !chains.empty() && !chains[0].curves.empty()) {
        const std::size_t len = chains[0].curves[0].size();
        std::vector<std::vector<double>> block_means;
        for (const Chain& ch : chains) {
            const std::size_t nb = R > 1 ? 1 : static_cast<std::size_t>(std::max(1, t.blocks));
            const std::size_t per = std::max<std::size_t>(1, ch.curves.size() / nb);
            for (std::size_t b = 0; b < nb && b * per < ch.curves.size(); ++b) {
                std::vector<double> mean(len, 0.0);
                const std::size_t end = std::min(ch.curves.size(), (b + 1) * per);
                for (std::size_t i = b * per; i < end; ++i)
                    for (std::size_t d = 0; d < len; ++d) mean[d] += ch.curves[i][d];
                for (double& v : mean) v /= static_cast<double>(end - b * per);   // sum first: exact for constant G
                block_means.push_back(mean);
            }
        }
        CsvWriter out(path_for(prefix, "correlation"), {"r", "G_mean", "G_err"});
        const double B = static_cast<double>(block_means.size());
        for (std::size_t d = 0; d < len; ++d) {
            double sum = 0.0, sum2 = 0.0;
            for (const auto& bm : block_means) { sum += bm[d]; sum2 += bm[d] * bm[d]; }
            const double mean = sum / B, sd = std::sqrt(std::max(0.0, sum2 / B - mean * mean));
            out.row(d, mean, B > 1 ? sd / std::sqrt(B - 1.0) : 0.0);
        }
    }

    // Histogram of the pooled samples over their range.
    const auto [lo_it, hi_it] = std::minmax_element(pooled.begin(), pooled.end());
    const double lo = pooled.empty() ? 0.0 : *lo_it, hi = pooled.empty() ? 0.0 : *hi_it;
    const std::vector<std::size_t> hist = stats::histogram(pooled, t.bins, lo, hi);
    const double width = (hi - lo) / static_cast<double>(hist.size());

    if (!prefix.empty()) {
        CsvWriter csv(path_for(prefix, "samples"), {"sample", "chain", "sweep", t.observable.c_str()});
        for (int c = 0; c < R; ++c) {
            const auto& m = chains[static_cast<std::size_t>(c)].st.m;
            for (std::size_t k = 0; k < m.size(); ++k)
                csv.row(k, c, t.burn_in + static_cast<long long>(k + 1) * t.interval, m[k]);
        }
        CsvWriter h(path_for(prefix, "histogram"), {"bin_low", "bin_high", "count", "density"});
        for (std::size_t b = 0; b < hist.size(); ++b)
            h.row(lo + width * b, lo + width * (b + 1), hist[b],
                  width > 0.0 ? static_cast<double>(hist[b]) / (static_cast<double>(pooled.size()) * width) : 0.0);
    }

    const double block = static_cast<double>(t.samples) / t.blocks;
    if (verbose) {
        std::cout << "Observable '" << t.observable << "': " << t.samples << " samples every " << t.interval
                  << " sweeps after " << t.burn_in << " sweeps of burn-in"
                  << (R > 1 ? ", in each of " + std::to_string(R) + " independent chains (errors: spread between chains)" : "")
                  << ".\n"
                  << "  mean                " << fmt(estimate[0]) << " +- " << fmt(error[0], 3) << '\n'
                  << "  |mean|              " << fmt(estimate[1]) << " +- " << fmt(error[1], 3) << '\n'
                  << "  N var(x)            " << fmt(estimate[2]) << " +- " << fmt(error[2], 3) << "   (susceptibility)\n"
                  << "  N (<x^2> - <|x|>^2) " << fmt(estimate[3]) << " +- " << fmt(error[3], 3) << '\n'
                  << "  Binder cumulant     " << fmt(estimate[4]) << " +- " << fmt(error[4], 3) << '\n'
                  << "  autocorrelation     " << fmt(tau, 3) << " samples (of |x|: " << fmt(tau_abs, 3) << ")"
                  << (R > 1 ? ", averaged over chains" : "") << '\n';
        if (thermal && !energy.empty())
            std::cout << "  energy per site     " << fmt(e_mean) << " +- " << fmt(e_err, 3) << '\n'
                      << "  specific heat       " << fmt(heat) << " +- " << fmt(heat_err, 3) << "   (N var(e) / T^2)\n";
        if (want_xi)
            std::cout << "  correlation length  " << fmt(xi) << " +- " << fmt(xi_err, 3) << "   (second moment; xi/L = "
                      << fmt(xi / ctx.side, 4) << ")\n";
        // A compact text histogram (at most 12 rows) to see one peak or two.
        const std::size_t rows = std::min<std::size_t>(12, hist.size());
        if (rows > 1 && width > 0.0) {
            std::vector<std::size_t> merged(rows, 0);
            for (std::size_t b = 0; b < hist.size(); ++b) merged[b * rows / hist.size()] += hist[b];
            const std::size_t peak = *std::max_element(merged.begin(), merged.end());
            std::cout << "  histogram of " << t.observable << ":\n";
            for (std::size_t r = 0; r < rows; ++r)
                std::cout << "    " << std::setw(10) << fmt(lo + (hi - lo) * (r + 0.5) / rows, 4) << " | "
                          << std::string(peak ? merged[r] * 40 / peak : 0, '#') << '\n';
        }
        // |mean|, the susceptibilities and the Binder cumulant depend on |x| and
        // x^2 only, so tau(|x|) governs their errors; tau(x) can be much longer
        // when the sign of x flips rarely, and then only the plain mean suffers.
        if (block < 10.0 * tau_abs)
            std::cout << "  WARNING: jackknife blocks of " << fmt(block, 3) << " samples are not much longer than the"
                      << " autocorrelation time; errors may be underestimated (use more --samples or a larger --interval).\n";
        else if (block < 10.0 * tau)
            std::cout << "  NOTE: x changes sign only rarely (autocorrelation " << fmt(tau, 3) << " samples), so the"
                      << " error of the plain mean is unreliable; |mean|, susceptibility and Binder are fine.\n";
        // Chains that disagree far beyond their own errors have not reached the
        // same stationary state (burn-in too short, or stuck in different
        // states). Checked on sign-independent quantities: the sign of x may
        // legitimately differ between chains in an ordered phase.
        if (R > 1 && (error[1] > 3.0 * error_within[1] || error[4] > 3.0 * error_within[4]))
            std::cout << "  WARNING: the chains disagree far more than their own errors allow (|mean| or Binder); they\n"
                         "  have probably not equilibrated -- use a longer --burn-in or more --samples.\n";
        if (absorbing)
            std::cout << "  NOTE: this model has an absorbing consensus; once reached, the samples stop changing.\n";
        if (!prefix.empty())
            std::cout << "Samples written to " << path_for(prefix, "samples") << ", histogram to "
                      << path_for(prefix, "histogram") << '\n';
    }
    return row;
}

// --------------------------------------------------------------------- sweep

struct SweepSpec {
    std::string key;
    std::vector<std::string> values;
};

inline SweepSpec parse_sweep(const std::string& text) {
    SweepSpec s;
    const std::size_t eq = text.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == text.size())
        throw OptionError("option --sweep: expected key=v1,v2,... or key=start:stop:step, got '" + text + "'");
    s.key = text.substr(0, eq);
    const std::string rest = text.substr(eq + 1);
    if (rest.find(':') != std::string::npos) {
        std::vector<double> parts;
        std::stringstream ss(rest);
        std::string item;
        while (std::getline(ss, item, ':')) {
            try { parts.push_back(std::stod(item)); }
            catch (const std::exception&) { throw OptionError("option --sweep: bad number '" + item + "'"); }
        }
        if (parts.size() != 3 || !(parts[2] > 0.0) || parts[1] < parts[0])
            throw OptionError("option --sweep: a range is start:stop:step with stop >= start and step > 0");
        const long long n = static_cast<long long>(std::floor((parts[1] - parts[0]) / parts[2] + 1e-9));
        if (n > 100000) throw OptionError("option --sweep: more than 100000 points");
        for (long long i = 0; i <= n; ++i) {
            std::ostringstream os;
            os << std::setprecision(12) << parts[0] + static_cast<double>(i) * parts[2];
            s.values.push_back(os.str());
        }
    } else {
        std::stringstream ss(rest);
        std::string item;
        while (std::getline(ss, item, ',')) if (!item.empty()) s.values.push_back(item);
    }
    if (s.values.empty()) throw OptionError("option --sweep: no values");
    return s;
}

} // namespace driver

#endif // ODSP_DRIVER_TASKS_HPP
