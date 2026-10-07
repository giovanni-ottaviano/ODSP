#ifndef ODSP_DRIVER_CATALOG_HPP
#define ODSP_DRIVER_CATALOG_HPP

#include <iostream>
#include <string>
#include <vector>

#include "Options.hpp"

// Everything the driver understands: models, graphs, tasks, initial
// conditions and observables, with their options. The help text is generated
// from these tables, and they are the dictionary for "did you mean"
// suggestions, so they are the single place to update when adding a model.

namespace driver {

struct OptionDoc {
    std::string name, value, fallback, help;   // fallback: default shown in the help, "required", or "" (none)
};

struct Entry {
    std::string            name;
    std::string            summary;
    std::vector<OptionDoc> options;
};

inline const std::vector<Entry>& model_catalog() {
    static const std::vector<Entry> models = {
        {"voter", "copy the opinion of a random neighbour (any number of opinions)",
         {{"update", "node|link|reverse", "node", "who copies whom on heterogeneous graphs"}}},
        {"rf-voter", "node-update voter model, rejection-free (fast when few links disagree)", {}},
        {"noisy-voter", "voter, or with probability noise a random opinion",
         {{"noise", "x", "required", "probability of a spontaneous random opinion"},
          {"opinions", "q", "2", "number of opinions (2 = +/-1, else 0..q-1)"}}},
        {"majority", "adopt the neighbourhood majority (plurality)",
         {{"tie", "random|keep", "random", "tie rule (keep = zero-temperature Glauber)"}}},
        {"qvoter", "adopt the opinion of q unanimous neighbours; nonconformity with prob p",
         {{"q", "n", "required", "panel size"},
          {"p", "x", "0", "nonconformity probability"},
          {"nonconformity", "independence|anticonformity|none", "independence", "kind of nonconformity"},
          {"repetition", "true|false", "true", "draw the panel with repetition"}}},
        {"nonlinear-voter", "flip with probability (fraction disagreeing)^alpha",
         {{"alpha", "x", "required", "nonlinearity (1 = voter)"}}},
        {"majority-vote", "majority with prob 1-noise, minority otherwise (Oliveira)",
         {{"noise", "x", "required", "probability of adopting the minority"}}},
        {"ising", "Ising model at temperature T",
         {{"temperature", "T", "required", "temperature"},
          {"J", "x", "1", "coupling"},
          {"h", "x", "0", "external field"},
          {"dynamics", "metropolis|glauber", "metropolis", "single-spin-flip dynamics"}}},
        {"potts", "q-state Potts model at temperature T (states 0..q-1)",
         {{"q", "n", "required", "number of states"},
          {"temperature", "T", "required", "temperature"},
          {"J", "x", "1", "coupling"},
          {"dynamics", "heat-bath|metropolis", "heat-bath", "single-site dynamics"}}},
        {"sznajd", "an agreeing pair of neighbours converts its other neighbours", {}},
        {"deffuant", "Deffuant-Weisbuch bounded confidence (continuous opinions)",
         {{"epsilon", "x", "required", "confidence bound"},
          {"mu", "x", "0.5", "convergence parameter, in (0, 1/2]"},
          {"tolerance", "x", "1e-06", "consensus / convergence tolerance"}}},
        {"hk", "Hegselmann-Krause bounded confidence (continuous opinions)",
         {{"epsilon", "x", "required", "confidence bound"},
          {"synchronous", "true|false", "true", "all agents update together (one round per sweep)"},
          {"tolerance", "x", "1e-09", "consensus / convergence tolerance"}}},
    };

    return models;
}

inline const std::vector<Entry>& graph_catalog() {
    static const std::vector<Entry> graphs = {
        {"lattice", "hypercubic lattice", {{"dims", "LxL[x...]", "32x32", "side lengths"},
                                           {"boundary", "periodic|open", "periodic", "boundary conditions"}}},
        {"complete", "complete graph (mean field)", {{"n", "N", "required", "nodes"}}},
        {"gnp", "Erdos-Renyi G(n, p)", {{"n", "N", "required", "nodes"}, {"edge-prob", "p", "required", "edge probability"}}},
        {"gnm", "Erdos-Renyi G(n, m)", {{"n", "N", "required", "nodes"}, {"edges", "M", "required", "number of edges"}}},
        {"regular", "random k-regular graph", {{"n", "N", "required", "nodes"}, {"k", "k", "required", "degree"}}},
        {"ws", "Watts-Strogatz small world", {{"n", "N", "required", "nodes"}, {"k", "k", "required", "ring degree (even)"},
                                               {"beta", "x", "required", "rewiring probability"}}},
        {"ba", "Barabasi-Albert scale-free", {{"n", "N", "required", "nodes"}, {"m", "m", "required", "links per new node"}}},
        {"star", "star: hub plus n-1 leaves", {{"n", "N", "required", "nodes"}}},
        {"file", "edge-list file (SNAP/KONECT/CSV-like)", {{"path", "file", "required", "edge-list file"},
                                                          {"lcc", "true|false", "true", "keep only the largest connected component"}}},
    };

    return graphs;
}

inline const std::vector<OptionDoc>& general_options() {
    static const std::vector<OptionDoc> opts = {
        {"model", "name", "required", "model to simulate (see --help models)"},
        {"graph", "name", "lattice", "topology (see --help graphs)"},
        {"task", "run|ensemble|stationary", "run", "what to compute (see --help tasks)"},
        {"seed", "n", "random", "master seed; the value used is printed and saved"},
        {"threads", "n", "0", "threads for ensembles (0 = all cores, at most $ODSP_MAX_THREADS if set); results do not depend on it"},
        {"csv", "prefix", "", "write results to <prefix>_*.csv and the configuration to <prefix>_config.txt"},
        {"config", "file", "", "read options from a file of 'key = value' lines (command line overrides it)"},
        {"regenerate-graph", "true|false", "false", "draw a new random network for every replica (default: one network for all)"},
        {"provenance", "true|false", "true", "start every output file with '#' lines: version, commit, date, command, configuration"},
        {"version", "", "", "print the ODSP version and git commit"},
    };

    return opts;
}

inline const std::vector<OptionDoc>& task_options() {
    static const std::vector<OptionDoc> opts = {
        {"steps", "n", "1000 (run) / 10000 (ensemble)", "run: sweeps to simulate; ensemble: cap per replica"},
        {"observables", "a,b,...", "model-dependent", "run: time series; ensemble: measured at the end"},
        {"every", "n", "1 (run)", "run/ensemble: record every n time units (ensemble: writes the replica-averaged time series)"},
        {"log-times", "[k]", "", "run/ensemble: record at log-spaced times, k per decade (bare flag: 10)"},
        {"stop", "true|false", "true", "run: stop when the dynamics is absorbed"},
        {"replicas", "n", "100 / 1", "ensemble: independent runs; stationary: independent chains, pooled (errors from their spread)"},
        {"time-unit", "sweeps|updates", "sweeps", "ensemble: unit of consensus times (and of --steps)"},
        {"observable", "name", "model-dependent", "stationary: the sampled observable"},
        {"burn-in", "n", "1000", "stationary: sweeps discarded first"},
        {"samples", "n", "1000", "stationary: number of samples"},
        {"interval", "n", "1", "stationary: sweeps between samples"},
        {"blocks", "n", "20", "stationary: jackknife blocks for the error bars"},
        {"bins", "n", "50", "stationary: bins of the histogram of the sampled observable"},
        {"threshold", "x", "", "ensemble: also record when the largest opinion share first reaches x (discrete)"},
        {"tw", "t1,t2,...", "", "run/ensemble: waiting times of the autocorrelation/overlap observables (added to the recording times)"},
        {"correlation", "true|false", "false", "write the spatial correlation G(r) to prefix_correlation.csv (lattices; ensembles need a time series)"},
        {"sweep", "key=v1,v2,... | key=start:stop:step", "", "repeat the ensemble or stationary task for each value of a parameter"},
    };

    return opts;
}

inline const std::vector<OptionDoc>& init_options() {
    static const std::vector<OptionDoc> opts = {
        {"init", "random|all|up|down|balanced|fractions|distinct|uniform", "model-dependent", "initial opinions"},
        {"up", "f", "0.5", "random: fraction of +1 (the rest -1)"},
        {"value", "x", "1", "all: the opinion every node starts with (up = +1, down = -1)"},
        {"opinions", "q", "model-dependent", "balanced: opinions 0..q-1 in equal shares"},
        {"fractions", "f0,f1,...", "", "fractions: share of each opinion 0, 1, ... (required with --init fractions)"},
        {"lo", "x", "0", "uniform: lower end"},
        {"hi", "x", "1", "uniform: upper end"},
        {"zealots", "count:opinion,...", "", "frozen nodes, e.g. 5:1,5:-1"},
        {"zealot-placement", "random|hubs", "random", "zealots on random nodes or on the highest-degree ones"},
    };

    return opts;
}

inline const std::vector<OptionDoc>& observable_docs() {
    static const std::vector<OptionDoc> obs = {
        {"m", "", "", "magnetization per site (+/-1 models); mean opinion (continuous)"},
        {"abs_m", "", "", "|m|"},
        {"mean", "", "", "mean opinion (= m)"},
        {"omega", "", "", "degree-weighted magnetization"},
        {"rho", "", "", "density of links joining different opinions"},
        {"opinions", "", "", "number of opinions still held (discrete)"},
        {"largest", "", "", "share of the largest opinion group (discrete)"},
        {"energy", "", "", "energy per site (ising, potts)"},
        {"order", "", "", "Potts order parameter (q f_max - 1)/(q - 1)"},
        {"range", "", "", "max - min opinion (continuous)"},
        {"clusters", "", "", "opinion clusters with >= 1% of the nodes (continuous)"},
        {"largest_cluster", "", "", "share of the largest opinion cluster (continuous)"},
        {"sweeps", "", "", "sweeps done (e.g. as an ensemble's final observable)"},
        {"length", "", "", "coarsening length: where the spatial correlation G(r) drops to 1/2 (lattices)"},
        {"autocorrelation", "", "", "two-time C(t, t_w): (1/N) sum s_i(t) s_i(t_w) for +/-1, Pearson for continuous; one column per --tw (time series)"},
        {"overlap", "", "", "fraction of nodes with the same opinion as at t_w; one column per --tw (time series, discrete)"},
        {"persistence", "", "", "fraction of nodes that never changed opinion"},
        {"activity", "", "", "fraction of attempted updates that changed an opinion (per recording interval; whole run as a final value)"},
        {"entropy", "", "", "Shannon entropy of the opinion shares (discrete)"},
        {"effective_opinions", "", "", "effective number of opinions exp(entropy) (discrete)"},
        {"segregation", "", "", "1 - rho / rho under random placement: 0 = mixed, 1 = segregated (discrete; 1 at consensus)"},
        {"variance", "", "", "variance of the opinions"},
        {"bimodality", "", "", "Sarle's bimodality coefficient of the opinions (> 5/9: two peaks)"},
    };

    return obs;
}

// Every option name the driver knows, for suggestions.
inline std::vector<std::string> all_option_names() {
    std::vector<std::string> names;
    auto add = [&names](const std::vector<OptionDoc>& docs) { for (const auto& d : docs) names.push_back(d.name); };
    add(general_options());
    add(task_options());
    add(init_options());
    for (const auto& m : model_catalog()) add(m.options);
    for (const auto& g : graph_catalog()) add(g.options);

    return names;
}

inline std::string suggestion(const std::string& word, const std::vector<std::string>& dictionary) {
    std::string best;
    std::size_t best_d = 3;   // suggest only close matches
    for (const std::string& w : dictionary) {
        const std::size_t d = edit_distance(word, w);
        if (d < best_d) { best_d = d; best = w; }
    }

    return best;
}

// as_options: print names as "--name" (options) or bare (observables).
inline void print_options(const std::vector<OptionDoc>& docs, const std::string& indent = "  ", bool as_options = true) {
    for (const auto& d : docs) {
        std::string flag = (as_options ? "--" : "") + d.name + (d.value.empty() ? "" : " " + d.value);
        std::cout << indent << flag;
        if (flag.size() < 34)
            std::cout << std::string(34 - flag.size(), ' ');
        else
            std::cout << "\n" << indent << std::string(34, ' ');

        std::cout << d.help;
        if (d.fallback == "required")
            std::cout << " (required)";
        else if (!d.fallback.empty())
            std::cout << " [" << d.fallback << "]";

        std::cout << '\n';
    }
}

inline void print_entries(const std::vector<Entry>& entries) {
    for (const auto& e : entries) {
        std::cout << "  " << e.name << ": " << e.summary << '\n';
        print_options(e.options, "      ");
    }
}

inline void print_help(const std::string& topic) {
    if (topic == "models") {
        std::cout << "Models (--model NAME) and their options:\n";
        print_entries(model_catalog());
    }
    else if (topic == "graphs") {
        std::cout << "Graphs (--graph NAME) and their options:\n";
        print_entries(graph_catalog());
        std::cout << "  Random networks are generated once from the seed and shared by all replicas\n"
                     "  (unless --regenerate-graph); sweeps reuse the same network at every value.\n";
    }
    else if (topic == "tasks") {
        std::cout << "Tasks (--task NAME):\n"
                     "  run         one trajectory; time series of --observables every --every sweeps\n"
                     "  ensemble    --replicas independent runs: consensus times, fixation probabilities,\n"
                     "              final values of --observables; with --every or --log-times also the\n"
                     "              time series averaged over replicas (absorbed runs keep their final value)\n"
                     "  stationary  burn-in, then --samples of --observable: mean, |mean|, susceptibility,\n"
                     "              Binder cumulant with jackknife errors, autocorrelation time\n"
                     "  --sweep KEY=VALUES repeats ensemble or stationary for each value and writes one table;\n"
                     "  points run in parallel within --threads (results do not depend on the thread count).\n\n";
        print_options(task_options());
    }
    else if (topic == "init") {
        std::cout << "Initial conditions and zealots:\n";
        print_options(init_options());
        std::cout << "  Defaults: random (+/-1, half up) for binary models, balanced q for potts and\n"
                     "  noisy-voter with --opinions > 2, uniform in [0,1] for deffuant and hk.\n";
    }
    else if (topic == "observables") {
        std::cout << "Observables (--observables a,b,... or --observable name):\n";
        print_options(observable_docs(), "  ", /*as_options=*/false);
    }
    else {
        std::cout <<
            "odsp - opinion dynamics simulations from the command line\n\n"
            "Usage: odsp --model NAME [--graph NAME ...] [--task run|ensemble|stationary] [options]\n"
            "       odsp --config FILE [options]\n\n"
            "Examples:\n"
            "  odsp --model voter --graph lattice --dims 64x64 --steps 2000 --csv out/voter\n"
            "  odsp --model qvoter --q 3 --p 0.2 --graph ba --n 2000 --m 3 --task stationary --samples 2000 --interval 10\n"
            "  odsp --model ising --temperature 2.3 --init up --task stationary --samples 2000 --interval 50\n"
            "  odsp --model majority-vote --init up --dims 16x16 --task stationary --samples 4000 --interval 10 --sweep noise=0.06:0.09:0.005 --csv mv\n"
            "  odsp --model deffuant --epsilon 0.2 --graph complete --n 1000 --task ensemble --observables clusters\n\n"
            "General options:\n";
        print_options(general_options());
        std::cout << "\nMore help: --help models | graphs | tasks | init | observables\n";
    }
}

} // namespace driver

#endif // ODSP_DRIVER_CATALOG_HPP
