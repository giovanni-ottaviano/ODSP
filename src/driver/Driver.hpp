#ifndef ODSP_DRIVER_DRIVER_HPP
#define ODSP_DRIVER_DRIVER_HPP

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "CsvWriter.hpp"
#include "Version.hpp"
#include "Catalog.hpp"
#include "Options.hpp"
#include "Parallel.hpp"
#include "Setup.hpp"
#include "Tasks.hpp"

// The odsp driver: options in, simulation out. run_driver() is the whole
// program (main() only forwards argv), so the tests can call it directly.

namespace driver {

// Options given but never read are errors: unknown keys get a suggestion,
// known ones are reported as not applying to this configuration.
inline void check_unused(const Options& o) {
    const std::vector<std::string> unused = o.unused();
    if (unused.empty()) return;
    const std::vector<std::string> known = all_option_names();
    std::string msg;

    for (const std::string& key : unused) {
        msg += "\n  --" + key;
        if (std::find(known.begin(), known.end(), key) != known.end()) {
            msg += " does not apply here (wrong model, graph, init or task?)";
        }
        else {
            const std::string s = suggestion(key, known);
            msg += " is not an option" + (s.empty() ? std::string() : " (did you mean --" + s + "?)");
        }
    }
    throw OptionError("unused options:" + msg + "\nSee --help.");
}

// The provenance written at the top of every output file: version and commit,
// date, the command line, and the complete effective configuration.
inline std::string provenance_text(const std::string& command, const Options& o, const std::string& prefix) {
    std::string text = odsp_version_string() + ", written " + iso_timestamp() + "\n" + "command: " + command + "\n" +
                       "configuration" + (prefix.empty() ? std::string() : " (replay: odsp --config " + prefix + "_config.txt)") + ":";
    std::istringstream config(o.effective_config());
    std::string line;
    while (std::getline(config, line))
        text += "\n  " + line;

    return text;
}

// Restores the previous CSV preamble on scope exit, so a driver run inside a
// longer program (or the tests) leaves no trace.
struct ScopedPreamble {
    std::string saved = CsvWriter::preamble();
    explicit ScopedPreamble(const std::string& text) { CsvWriter::set_preamble(text); }
    ~ScopedPreamble() { CsvWriter::set_preamble(saved); }
};

inline void save_config(const Options& o, const std::string& prefix) {
    if (prefix.empty()) return;
    const std::string path = prefix + "_config.txt";
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);

    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot write '" + path + "'");
    out << "# " << odsp_version_string() << ", written " << iso_timestamp() << "\n"
        << "# odsp configuration -- rerun exactly with: odsp --config " << path << "\n" << o.effective_config();
}

// Everything read for one configuration (one point of a sweep).
template <typename T>
struct Plan {
    unsigned    seed = 0;
    int         threads = 0;
    std::string prefix;
    std::string task;
    bool        provenance = true;
    Setup<T>    setup;
    TaskParams  params;
};

template <typename T>
Plan<T> read_plan(Options& o, TopologyCache* cache = nullptr) {
    Plan<T> p;
    const long long seed = o.integer("seed", 0);   // always set by run_driver
    if (seed < 0 || seed > 0xFFFFFFFFLL)
        throw OptionError("option --seed must be in 0..4294967295 (got " + std::to_string(seed) + ")");

    p.seed = static_cast<unsigned>(seed);
    p.threads = static_cast<int>(o.integer("threads", 0));
    p.prefix = o.str("csv", "");
    p.provenance = o.flag("provenance", true);
    p.task = o.choice("task", "run", {"run", "ensemble", "stationary"});
    o.str("sweep", "");   // read (and so recorded) here; interpreted by execute()
    std::mt19937 seeder(p.seed);
    p.setup = read_setup<T>(o, seeder(), cache);
    p.params = read_task(o, p.task, p.setup.model, p.setup.init);

    // Fail now rather than after a long run: observables must exist for this
    // model and graph, and the model must accept its parameters. In `run` and
    // in ensemble time series, observables may follow the trajectory.
    ObservableContext& ctx = p.setup.context;
    ctx.sweeps = p.task != "ensemble" || p.params.sweeps_unit;
    const bool series = p.task == "run" || (p.task == "ensemble" && !p.params.times.empty());
    for (const auto& name : p.params.observables) make_observable<T>(name, p.setup.model, series, ctx);
    if (p.task == "stationary") {
        if (is_two_time(p.params.observable))
            throw OptionError("the stationary task samples one observable at a time; two-time observables are for run/ensemble");
        make_observable<T>(p.params.observable, p.setup.model, true, ctx);
    }
    if (p.params.correlation && !ctx.lattice) throw OptionError("--correlation needs a lattice (--graph lattice)");
    std::mt19937 scratch(0);
    try {
        p.setup.factory(scratch);
    }
    catch (const std::invalid_argument& e) {
        throw OptionError(e.what());
    }
    return p;
}

// The columns a sweep prints (all columns go to the CSV).
inline bool printed_in_sweep(const std::string& task, const std::string& column) {
    if (task == "ensemble")
        return column == "consensus_fraction" || column == "mean_consensus_time" || column == "median_consensus_time" ||
               column == "absorbed_fraction" || column == "exit_prob_up" ||
               (column.rfind("mean_", 0) == 0 && column != "mean_consensus_time_lower_bound");

    return column == "abs_mean" || column == "abs_mean_err" || column == "abs_susceptibility" || column == "binder" ||
           column == "binder_err" || column == "tau_abs";
}

template <typename T>
void execute(const Options& raw, const std::string& command) {
    Options o = raw;
    const std::string sweep_text = o.str("sweep", "");
    if (sweep_text.empty()) {
        Plan<T> p = read_plan<T>(o);
        check_unused(o);
        const ScopedPreamble preamble(p.provenance ? provenance_text(command, o, p.prefix) : std::string());
        std::cout << "odsp: " << p.setup.description << "\n"
                  << "task: " << p.task << ", seed " << p.seed << "\n\n";
        save_config(o, p.prefix);
        std::mt19937 seeder(p.seed);
        seeder();   // the graph seed, used in read_plan
        const unsigned task_seed = seeder();
        if (p.task == "run")
            task_run<T>(p.params, p.setup, task_seed, p.prefix);
        else if (p.task == "ensemble")
            task_ensemble<T>(p.params, p.setup, task_seed, p.threads, p.prefix, true);
        else
            task_stationary<T>(p.params, p.setup, task_seed, p.prefix, true, resolve_threads(p.threads));
        if (!p.prefix.empty())
            std::cout << "Configuration saved to " << p.prefix << "_config.txt\n";

        return;
    }

    // ---- sweep: the same task for each value of one option ----
    const SweepSpec sweep = parse_sweep(sweep_text);
    for (const char* reserved : {"seed", "task", "sweep", "csv", "threads", "config"})
        if (sweep.key == reserved)
            throw OptionError("option --sweep: cannot sweep --" + sweep.key);

    // Validate every point before running
    // Plans share built networks through the cache, so keeping them all costs one network, not one per point.
    std::vector<Plan<T>> plans;
    TopologyCache cache;
    Options first;
    for (const std::string& value : sweep.values) {
        Options point = raw;
        point.set(sweep.key, value);
        plans.push_back(read_plan<T>(point, &cache));

        if (plans.size() == 1) {
            check_unused(point);
            if (!point.was_read(sweep.key))
                throw OptionError("option --sweep: --" + sweep.key + " does not apply to this configuration");

            first = point;
        }
    }
    const Plan<T>& p0 = plans.front();
    if (p0.task == "run")
        throw OptionError("--sweep needs --task ensemble or --task stationary");
    const ScopedPreamble preamble(p0.provenance ? provenance_text(command, first, p0.prefix) : std::string());

    std::cout << "odsp: " << p0.setup.description << "\n"
              << "task: " << p0.task << " for " << sweep.values.size() << " values of --" << sweep.key
              << ", seed " << p0.seed << "\n\n";

    save_config(first, p0.prefix);

    // Points run in parallel. Seeds are drawn up front in point order, so the
    // results do not depend on the thread count.
    // P points at a time, each with threads / P threads for its replicas or chains.
    std::mt19937 seeder(p0.seed);
    seeder();   // the graph seed
    const std::size_t n = plans.size();
    std::vector<unsigned> point_seeds(n);
    for (unsigned& sd : point_seeds) sd = seeder();
    const int budget = resolve_threads(p0.threads);
    const int concurrent = std::max(1, std::min(budget, static_cast<int>(n)));
    const int inner = std::max(1, budget / concurrent);

    // Rows are printed and written in point order as soon as all earlier  points are done.
    std::unique_ptr<CsvWriter> csv;
    std::vector<std::size_t> widths;   // printed width of each column
    std::vector<std::string> poorly_sampled;
    std::vector<Row> rows(n);
    std::vector<char> finished(n, 0);
    std::size_t next_to_print = 0;
    std::mutex print_mutex;
    auto print_ready_rows = [&]() {   // called with print_mutex held
        for (; next_to_print < n && finished[next_to_print]; ++next_to_print) {
            const std::size_t i = next_to_print;
            const Plan<T>& p = plans[i];
            const Row& row = rows[i];
            if (i == 0) {
                std::vector<std::string> header = {sweep.key};
                for (const auto& c : row)
                    header.push_back(c.first);

                csv = std::make_unique<CsvWriter>(p.prefix.empty() ? std::string() : p.prefix + "_sweep.csv", header);
                std::cout << std::setw(12) << sweep.key;
                for (const auto& c : row) {
                    widths.push_back(std::max<std::size_t>(14, c.first.size() + 2));
                    if (printed_in_sweep(p.task, c.first))
                        std::cout << std::setw(static_cast<int>(widths.back())) << c.first;
                }
                std::cout << '\n';
            }
            std::vector<double> values;
            for (const auto& c : row)
                values.push_back(c.second);

            csv->row_values(sweep.values[i], values);
            std::cout << std::setw(12) << sweep.values[i];
            for (std::size_t k = 0; k < row.size(); ++k)
                if (printed_in_sweep(p.task, row[k].first))
                    std::cout << std::setw(static_cast<int>(widths[k])) << fmt(row[k].second);

            std::cout << std::endl;
            if (p.task == "stationary") {   // same check as a single stationary run
                double tau = 0.0;   // tau(|x|) governs the printed columns (see task_stationary)
                for (const auto& c : row)
                    if (c.first == "tau_abs")
                        tau = c.second;
                if (static_cast<double>(p.params.samples) / p.params.blocks < 10.0 * tau)
                    poorly_sampled.push_back(sweep.values[i]);
            }
        }
    };
    parallel_for(static_cast<int>(n), concurrent, [&](int idx) {
        const std::size_t i = static_cast<std::size_t>(idx);
        const Plan<T>& p = plans[i];
        const std::string point_prefix = p.prefix.empty() ? std::string() : p.prefix + "_point" + std::to_string(i);
        Row row = p.task == "ensemble" ? task_ensemble<T>(p.params, p.setup, point_seeds[i], inner, point_prefix, false)
                                       : task_stationary<T>(p.params, p.setup, point_seeds[i], "", false, inner);
        std::lock_guard<std::mutex> lock(print_mutex);
        rows[i] = std::move(row);
        finished[i] = 1;
        print_ready_rows();
    });
    if (!poorly_sampled.empty()) {
        std::cout << "\nWARNING: at --" << sweep.key << " =";
        for (const auto& v : poorly_sampled)
            std::cout << " " << v;
        std::cout << " the jackknife blocks are not much longer than the autocorrelation time (tau);\n"
                     "the error bars there may be underestimated (use more --samples or a larger --interval).\n";
    }
    if (!p0.prefix.empty())
        std::cout << "\nSweep table written to " << p0.prefix << "_sweep.csv"
                  << (p0.task == "ensemble" ? " (per-point replica data in " + p0.prefix + "_point<i>_*.csv)" : "")
                  << "; configuration saved to " << p0.prefix << "_config.txt\n";
}

// The whole program. Returns the exit status: 0 ok, 1 runtime failure, 2 bad options
inline int run_driver(int argc, char** argv) {
    try {
        Options raw = Options::from_args(argc, argv);
        if (raw.has("version")) {
            std::cout << odsp_version_string() << '\n';
            return 0;
        }
        if (argc <= 1 || raw.has("help")) {
            std::string topic = raw.has("help") ? raw.str("help") : "";
            print_help(topic == "true" ? "" : topic);
            return 0;
        }
        if (!raw.has("model"))
            throw OptionError("missing required option --model (see --help models)");
        if (!raw.has("seed"))
            raw.set("seed", std::to_string(std::random_device{}()));
        Options probe = raw;
        const std::string model = probe.str("model");
        check_name("model", model, model_catalog());
        // The command line, shell-quoted where needed, for the provenance.
        std::string command = "odsp";
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const bool plain = !a.empty() && a.find_first_of(" \t'\"\\$*?;&|<>()") == std::string::npos;
            command += " " + (plain ? a : "'" + a + "'");
        }
        if (is_continuous(model)) execute<double>(raw, command);
        else                      execute<int>(raw, command);
        return 0;
    } catch (const OptionError& e) {
        std::cerr << "odsp: error: " << e.what() << '\n';
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "odsp: error: " << e.what() << '\n';
        return 1;
    }
}

} // namespace driver

#endif // ODSP_DRIVER_DRIVER_HPP
