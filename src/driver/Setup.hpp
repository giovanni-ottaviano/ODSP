#ifndef ODSP_DRIVER_SETUP_HPP
#define ODSP_DRIVER_SETUP_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "AdjacencyGraph.hpp"
#include "CompleteGraph.hpp"
#include "EdgeList.hpp"
#include "EdgeListIO.hpp"
#include "Generators.hpp"
#include "RegularLattice.hpp"
#include "Ensemble.hpp"
#include "OpinionClusters.hpp"
#include "Utils.hpp"
#include "VoterModel.hpp"
#include "RejectionFreeVoterModel.hpp"
#include "NoisyVoterModel.hpp"
#include "MajorityModel.hpp"
#include "QVoterModel.hpp"
#include "NonlinearVoterModel.hpp"
#include "MajorityVoteModel.hpp"
#include "IsingModel.hpp"
#include "PottsModel.hpp"
#include "SznajdModel.hpp"
#include "DeffuantModel.hpp"
#include "HegselmannKrauseModel.hpp"

#include "Catalog.hpp"
#include "Options.hpp"

// Turning options into a model factory: topology, initial opinions, zealots,
// model parameters, and named observables. Each reader consumes exactly the
// options that apply, so anything left over is a mistake the driver reports.

namespace driver {

inline bool is_continuous(const std::string& model) { return model == "deffuant" || model == "hk"; }

inline bool is_binary_only(const std::string& model) {
    return model == "qvoter" || model == "nonlinear-voter" || model == "majority-vote" || model == "ising" ||
           model == "sznajd";
}

inline void check_name(const std::string& kind, const std::string& name, const std::vector<Entry>& catalog) {
    std::vector<std::string> names;
    for (const auto& e : catalog)
        names.push_back(e.name);
    if (std::find(names.begin(), names.end(), name) != names.end())
        return;

    const std::string s = suggestion(name, names);
    throw OptionError("unknown " + kind + " '" + name + "'" + (s.empty() ? "" : " (did you mean '" + s + "'?)") +
                      "; see --help " + kind + "s");
}

// ------------------------------------------------------------------ topology

struct TopologySpec {
    std::string kind;
    std::size_t n = 0;
    std::vector<int> dims;
    Boundary boundary = Boundary::Periodic;
    double edge_prob = 0.0, beta = 0.0;
    std::size_t edges = 0, k = 0, m = 0;
    std::string path;
    bool lcc = true;
    bool random = false;            // a random network (can be regenerated per replica)
};

struct Topology {
    std::size_t nodes = 0;
    bool        lattice = false, complete = false;
    std::vector<int> dims;
    Boundary    boundary = Boundary::Periodic;
    EdgeList    edges;
    std::string description;
};

inline std::vector<int> parse_dims(const std::string& s) {
    std::vector<int> dims;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, 'x')) {
        try {
            std::size_t used = 0;
            const int d = std::stoi(part, &used);
            if (used != part.size() || d < 1)
                throw std::invalid_argument(part);
            dims.push_back(d);
        }
        catch (const std::exception&) {
            throw OptionError("option --dims: expected sides like 32x32, got '" + s + "'");
        }
    }
    if (dims.empty())
        throw OptionError("option --dims: expected sides like 32x32, got '" + s + "'");

    return dims;
}

inline TopologySpec read_topology(Options& o) {
    TopologySpec t;
    t.kind = o.str("graph", "lattice");
    check_name("graph", t.kind, graph_catalog());
    auto count = [&o](const std::string& key) {
        const long long v = o.integer(key);
        if (v < 1)
            throw OptionError("option --" + key + " must be >= 1");
        return static_cast<std::size_t>(v);
    };
    if (t.kind == "lattice") {
        t.dims = parse_dims(o.str("dims", "32x32"));
        t.boundary = o.choice("boundary", "periodic", {"periodic", "open"}) == "open" ? Boundary::Open : Boundary::Periodic;
    }
    else if (t.kind == "complete" || t.kind == "star") {
        t.n = count("n");
    }
    else if (t.kind == "gnp") {
        t.n = count("n"); t.edge_prob = o.num("edge-prob"); t.random = true;
    }
    else if (t.kind == "gnm") {
        t.n = count("n"); t.edges = static_cast<std::size_t>(o.integer("edges")); t.random = true;
    }
    else if (t.kind == "regular") {
        t.n = count("n"); t.k = static_cast<std::size_t>(o.integer("k")); t.random = true;
    }
    else if (t.kind == "ws") {
        t.n = count("n"); t.k = static_cast<std::size_t>(o.integer("k")); t.beta = o.num("beta"); t.random = true;
    }
    else if (t.kind == "ba") {
        t.n = count("n"); t.m = count("m"); t.random = true;
    }
    else if (t.kind == "file") {
        t.path = o.str("path"); t.lcc = o.flag("lcc", true);
    }
    return t;
}

inline Topology build_topology(const TopologySpec& t, std::mt19937& rng) {
    Topology g;
    std::ostringstream d;
    if (t.kind == "lattice") {
        g.lattice = true;
        g.dims = t.dims;
        g.boundary = t.boundary;
        g.nodes = 1;
        for (int s : t.dims)
            g.nodes *= static_cast<std::size_t>(s);
        d << "lattice ";
        for (std::size_t i = 0; i < t.dims.size(); ++i)
            d << (i ? "x" : "") << t.dims[i];

        d << (t.boundary == Boundary::Open ? " open" : " periodic");
    }
    else if (t.kind == "complete") {
        g.complete = true; g.nodes = t.n; d << "complete graph";
    }
    else if (t.kind == "file") {
        const EdgeListFile f = read_edge_list(t.path);
        if (t.lcc) {
            const Subgraph s = largest_connected_component(f.nodes, f.edges);
            g.nodes = s.nodes; g.edges = s.edges;
        }
        else {
            g.nodes = f.nodes; g.edges = f.edges;
        }
        d << "network from " << t.path << (t.lcc ? " (largest component)" : "");
    }
    else {
        g.nodes = t.n;
        try {
            if (t.kind == "star")         { g.edges = star_edges(t.n - 1);                     d << "star"; }
            else if (t.kind == "gnp")     { g.edges = gnp_edges(t.n, t.edge_prob, rng);        d << "G(n,p) p=" << t.edge_prob; }
            else if (t.kind == "gnm")     { g.edges = gnm_edges(t.n, t.edges, rng);            d << "G(n,m) m=" << t.edges; }
            else if (t.kind == "regular") { g.edges = random_regular_edges(t.n, t.k, rng);     d << t.k << "-regular random graph"; }
            else if (t.kind == "ws")      { g.edges = watts_strogatz_edges(t.n, t.k, t.beta, rng); d << "Watts-Strogatz k=" << t.k << " beta=" << t.beta; }
            else if (t.kind == "ba")      { g.edges = barabasi_albert_edges(t.n, t.m, rng);    d << "Barabasi-Albert m=" << t.m; }
        }
        catch (const std::invalid_argument& e) {
            throw OptionError(std::string("graph ") + t.kind + ": " + e.what());
        }
    }
    d << ", N = " << g.nodes;
    if (!g.lattice && !g.complete)
        d << ", " << g.edges.size() << " edges";
    g.description = d.str();

    return g;
}

// Built networks, keyed by their parameters and graph seed, so a sweep reads a
// file or generates a network once
class TopologyCache {
public:
    std::shared_ptr<const Topology> get(const TopologySpec& t, unsigned graph_seed) {
        std::ostringstream key;
        key << t.kind << '|' << t.n << '|' << static_cast<int>(t.boundary) << '|' << t.edge_prob << '|' << t.beta << '|'
            << t.edges << '|' << t.k << '|' << t.m << '|' << t.path << '|' << t.lcc << '|' << graph_seed << '|';
        for (int d : t.dims)
            key << d << 'x';

        auto& slot = _built[key.str()];
        if (!slot) {
            std::mt19937 rng(graph_seed);
            slot = std::make_shared<const Topology>(build_topology(t, rng));
        }
        return slot;
    }

private:
    std::map<std::string, std::shared_ptr<const Topology>> _built;
};

template <typename T>
std::unique_ptr<BaseGraph<T>> make_graph(const Topology& g, std::vector<T> states) {
    if (g.lattice)
        return std::make_unique<RegularLattice<T>>(std::move(states), std::vector<int>(g.dims), g.boundary);
    if (g.complete)
        return std::make_unique<CompleteGraph<T>>(std::move(states));

    return std::make_unique<AdjacencyGraph<T>>(std::move(states), g.edges);
}

// ------------------------------------------------------------------- model

struct ModelSpec {
    std::string name;
    std::string update = "node", tie = "random", nonconformity = "independence";
    std::string ising_dynamics = "metropolis", potts_dynamics = "heat-bath";
    double noise = 0.0, p = 0.0, alpha = 1.0, temperature = 1.0, J = 1.0, h = 0.0;
    double epsilon = 0.0, mu = 0.5, tolerance = 1e-6;
    int q = 2, opinions = 2;
    bool repetition = true, synchronous = true;

    std::string description;
};

inline ModelSpec read_model(Options& o) {
    ModelSpec s;
    s.name = o.str("model");
    check_name("model", s.name, model_catalog());
    std::ostringstream d;
    d << s.name;
    auto positive_int = [&o](const std::string& key, long long fallback = -1) {
        const long long v = fallback < 0 ? o.integer(key) : o.integer(key, fallback);
        if (v < 1)
            throw OptionError("option --" + key + " must be >= 1");

        return static_cast<int>(v);
    };
    if (s.name == "voter") {
        s.update = o.choice("update", "node", {"node", "link", "reverse"});
        d << " (" << s.update << " update)";
    }
    else if (s.name == "noisy-voter") {
        s.noise = o.num("noise"); s.opinions = positive_int("opinions", 2);
        d << " noise=" << s.noise << (s.opinions > 2 ? " opinions=" + std::to_string(s.opinions) : "");
    }
    else if (s.name == "majority") {
        s.tie = o.choice("tie", "random", {"random", "keep"});
        d << " (tie: " << s.tie << ")";
    }
    else if (s.name == "qvoter") {
        s.q = positive_int("q"); s.p = o.num("p", 0.0);
        s.nonconformity = o.choice("nonconformity", "independence", {"independence", "anticonformity", "none"});
        s.repetition = o.flag("repetition", true);
        d << " q=" << s.q << " p=" << s.p << " (" << s.nonconformity << ")";
    }
    else if (s.name == "nonlinear-voter") {
        s.alpha = o.num("alpha");
        d << " alpha=" << s.alpha;
    }
    else if (s.name == "majority-vote") {
        s.noise = o.num("noise");
        d << " noise=" << s.noise;
    }
    else if (s.name == "ising") {
        s.temperature = o.num("temperature"); s.J = o.num("J", 1.0); s.h = o.num("h", 0.0);
        s.ising_dynamics = o.choice("dynamics", "metropolis", {"metropolis", "glauber"});
        d << " T=" << s.temperature << " J=" << s.J << " h=" << s.h << " (" << s.ising_dynamics << ")";
    }
    else if (s.name == "potts") {
        s.q = positive_int("q"); s.temperature = o.num("temperature"); s.J = o.num("J", 1.0);
        s.potts_dynamics = o.choice("dynamics", "heat-bath", {"heat-bath", "metropolis"});
        d << " q=" << s.q << " T=" << s.temperature << " (" << s.potts_dynamics << ")";
    }
    else if (s.name == "deffuant") {
        s.epsilon = o.num("epsilon"); s.mu = o.num("mu", 0.5); s.tolerance = o.num("tolerance", 1e-6);
        d << " epsilon=" << s.epsilon << " mu=" << s.mu;
    }
    else if (s.name == "hk") {
        s.epsilon = o.num("epsilon"); s.synchronous = o.flag("synchronous", true); s.tolerance = o.num("tolerance", 1e-9);
        d << " epsilon=" << s.epsilon << (s.synchronous ? " (synchronous)" : " (asynchronous)");
    }
    s.description = d.str();

    return s;
}

template <typename T>
std::unique_ptr<Model<T>> make_model(const ModelSpec& s, std::unique_ptr<BaseGraph<T>> g, std::mt19937* rng) {
    if constexpr (std::is_floating_point<T>::value) {
        if (s.name == "deffuant")
            return std::make_unique<DeffuantModel<T>>(std::move(g), rng, s.epsilon, s.mu, s.tolerance);

        return std::make_unique<HegselmannKrauseModel<T>>(std::move(g), rng, s.epsilon, s.synchronous, s.tolerance);
    }
    else {
        if (s.name == "voter") {
            const VoterUpdate u = s.update == "link" ? VoterUpdate::Link
                                : s.update == "reverse" ? VoterUpdate::Reverse : VoterUpdate::Node;
            return std::make_unique<VoterModel<T>>(std::move(g), rng, u);
        }
        if (s.name == "rf-voter")
            return std::make_unique<RejectionFreeVoterModel<T>>(std::move(g), rng);
        if (s.name == "noisy-voter") {
            std::vector<T> set;
            if (s.opinions == 2)
                set = {T(1), T(-1)};
            else
                for (int k = 0; k < s.opinions; ++k) set.push_back(T(k));

            return std::make_unique<NoisyVoterModel<T>>(std::move(g), rng, s.noise, set);
        }
        if (s.name == "majority")
            return std::make_unique<MajorityModel<T>>(std::move(g), rng, s.tie == "keep" ? TieRule::KeepCurrent : TieRule::Random);
        if (s.name == "qvoter") {
            const Nonconformity n = s.nonconformity == "anticonformity" ? Nonconformity::Anticonformity
                                  : s.nonconformity == "none" ? Nonconformity::None : Nonconformity::Independence;

            return std::make_unique<QVoterModel<T>>(std::move(g), rng, s.q, s.p, n, s.repetition);
        }
        if (s.name == "nonlinear-voter")
            return std::make_unique<NonlinearVoterModel<T>>(std::move(g), rng, s.alpha);
        if (s.name == "majority-vote")
            return std::make_unique<MajorityVoteModel<T>>(std::move(g), rng, s.noise);
        if (s.name == "ising")
            return std::make_unique<IsingModel<T>>(std::move(g), rng, s.temperature, s.J, s.h,
                                                   s.ising_dynamics == "glauber" ? IsingDynamics::Glauber : IsingDynamics::Metropolis);
        if (s.name == "potts")
            return std::make_unique<PottsModel<T>>(std::move(g), rng, s.q, s.temperature, s.J,
                                                   s.potts_dynamics == "metropolis" ? PottsDynamics::Metropolis : PottsDynamics::HeatBath);

        return std::make_unique<SznajdModel<T>>(std::move(g), rng);   // sznajd
    }
}

// --------------------------------------------------- initial opinions + zealots

// The opinions a model is defined for: any value, +/-1, or 0..q-1.
struct OpinionSet {
    enum Kind { Any, Binary, Range } kind = Any;
    int q = 0;

    bool contains(double v) const {
        if (kind == Binary)
            return v == 1.0 || v == -1.0;
        if (kind == Range)
            return v == std::floor(v) && v >= 0.0 && v < static_cast<double>(q);

        return true;
    }
    std::string describe() const {
        if (kind == Binary)
            return "+1 or -1";
        if (kind == Range)
            return "0.." + std::to_string(q - 1);

        return "any value";
    }
};

inline OpinionSet opinion_set(const ModelSpec& m) {
    OpinionSet set;
    if (is_binary_only(m.name))
        set.kind = OpinionSet::Binary;
    else if (m.name == "potts") {
        set.kind = OpinionSet::Range;
        set.q = m.q;
    }
    else if (m.name == "noisy-voter") {
        if (m.opinions == 2)
            set.kind = OpinionSet::Binary;
        else {
            set.kind = OpinionSet::Range;
            set.q = m.opinions;
        }
    }
    return set;
}

struct InitSpec {
    std::string kind;
    double up = 0.5, value = 1.0, lo = 0.0, hi = 1.0;
    int opinions = 2;
    std::vector<double> fractions;
    std::vector<std::pair<std::size_t, double>> zealots;   // (count, opinion)
    bool zealots_on_hubs = false;
    bool multi = false;   // more than the two opinions +/-1
    std::string description;
};

inline std::string fmt_opinion(double v) {
    std::ostringstream os;
    os << v;
    return os.str();
}

inline InitSpec read_init(Options& o, const ModelSpec& m) {
    InitSpec s;
    const bool continuous = is_continuous(m.name);
    std::string fallback = continuous ? "uniform" : "random";
    if (m.name == "potts" || (m.name == "noisy-voter" && m.opinions > 2))
        fallback = "balanced";
    s.kind = o.choice("init", fallback, {"random", "all", "up", "down", "balanced", "fractions", "distinct", "uniform"});
    std::ostringstream d;
    const std::string requested = s.kind;
    if (s.kind == "up")   { s.kind = "all"; s.value = 1.0; }
    if (s.kind == "down") { s.kind = "all"; s.value = -1.0; }

    if (s.kind == "random") {
        s.up = o.num("up", 0.5);
        d << "random +/-1, up fraction " << s.up;
    }
    else if (s.kind == "all") {
        if (requested == "all")
            s.value = o.num("value", 1.0);   // up/down fix the value
        d << "all nodes at " << s.value;
    }
    else if (s.kind == "balanced") {
        const int fallback_q = m.name == "potts" ? m.q : m.opinions;
        s.opinions = static_cast<int>(o.integer("opinions", fallback_q));
        if (s.opinions < 1)
            throw OptionError("option --opinions must be >= 1");
        s.multi = true;
        d << "opinions 0.." << s.opinions - 1 << " in equal shares";
    }
    else if (s.kind == "fractions") {
        s.fractions = o.num_list("fractions");
        s.multi = true;
        d << "opinions 0.." << s.fractions.size() - 1 << " in given shares";
    }
    else if (s.kind == "distinct") {
        s.multi = true;
        d << "every node its own opinion";
    }
    else if (s.kind == "uniform") {
        s.lo = o.num("lo", 0.0); s.hi = o.num("hi", 1.0);
        d << "uniform in [" << s.lo << ", " << s.hi << ")";
    }

    if (continuous && s.kind != "uniform" && s.kind != "all")
        throw OptionError("model " + m.name + " has continuous opinions: use --init uniform or all");
    if (!continuous && s.kind == "uniform")
        throw OptionError("--init uniform is for continuous-opinion models (deffuant, hk)");
    // Every opinion the start can produce must belong to the model's set.
    const OpinionSet allowed = opinion_set(m);
    if (allowed.kind != OpinionSet::Any) {
        if (s.kind == "distinct")
            throw OptionError("model " + m.name + " is defined for opinions " + allowed.describe() +
                              ", but --init distinct gives every node its own label");
        std::vector<double> produced;
        if (s.kind == "random")    produced = {1.0, -1.0};
        if (s.kind == "all")       produced = {s.value};
        if (s.kind == "balanced")  for (int k = 0; k < s.opinions; ++k) produced.push_back(k);
        if (s.kind == "fractions") for (std::size_t k = 0; k < s.fractions.size(); ++k) produced.push_back(static_cast<double>(k));
        for (double v : produced)
            if (!allowed.contains(v))
                throw OptionError("model " + m.name + " is defined for opinions " + allowed.describe() + ", but --init " +
                                  requested + " starts with opinion " + fmt_opinion(v));
    }

    for (const std::string& item : o.word_list("zealots", "")) {
        const std::size_t colon = item.find(':');
        try {
            if (colon == std::string::npos)
                throw std::invalid_argument(item);
            std::size_t used = 0;
            const long long count = std::stoll(item.substr(0, colon), &used);
            const double value = std::stod(item.substr(colon + 1));
            if (count < 0)
                throw std::invalid_argument(item);
            if (!continuous && value != std::floor(value))
                throw std::invalid_argument(item);

            s.zealots.emplace_back(static_cast<std::size_t>(count), value);
        } catch (const std::exception&) {
            throw OptionError("option --zealots: expected count:opinion items like 5:1,5:-1, got '" + item + "'");
        }
        if (!allowed.contains(s.zealots.back().second))
            throw OptionError("option --zealots: model " + m.name + " is defined for opinions " + allowed.describe() +
                              ", not " + fmt_opinion(s.zealots.back().second));
    }
    if (!s.zealots.empty()) {
        s.zealots_on_hubs = o.choice("zealot-placement", "random", {"random", "hubs"}) == "hubs";
        d << "; zealots";
        for (const auto& z : s.zealots)
            d << " " << z.first << "x" << z.second;
        d << (s.zealots_on_hubs ? " on hubs" : " at random");
    }
    s.description = d.str();
    return s;
}

template <typename T>
std::vector<T> make_states(const InitSpec& s, std::size_t n, std::mt19937& rng) {
    auto convert = [](const std::vector<int>& v) { return std::vector<T>(v.begin(), v.end()); };
    if (s.kind == "random")    return convert(create_random_lattice(n, s.up, rng));
    if (s.kind == "all")       return std::vector<T>(n, static_cast<T>(s.value));
    if (s.kind == "balanced")  return convert(create_balanced_opinions(n, s.opinions, rng));
    if (s.kind == "fractions") return convert(create_opinions_with_fractions(n, s.fractions, rng));
    if (s.kind == "distinct")  return convert(create_distinct_opinions(n));
    const std::vector<double> u = create_uniform_opinions(n, rng, s.lo, s.hi);   // uniform

    return std::vector<T>(u.begin(), u.end());
}

// Picks the zealot nodes, sets their opinions in the graph and returns them.
template <typename T>
std::vector<std::size_t> place_zealots(const InitSpec& s, BaseGraph<T>& g, std::mt19937& rng) {
    std::size_t total = 0;
    for (const auto& z : s.zealots) total += z.first;
    if (total == 0)
        return {};
    if (total > g.size())
        throw OptionError("more zealots than nodes");
    std::vector<std::size_t> order(g.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (s.zealots_on_hubs)
        std::stable_sort(order.begin(), order.end(), [&g](std::size_t a, std::size_t b) { return g.degree(a) > g.degree(b); });
    else
        std::shuffle(order.begin(), order.end(), rng);
    std::vector<std::size_t> chosen;
    std::size_t next = 0;
    for (const auto& z : s.zealots)
        for (std::size_t c = 0; c < z.first; ++c, ++next) {
            g.set_state(order[next], static_cast<T>(z.second));
            chosen.push_back(order[next]);
        }

    return chosen;
}

// --------------------------------------------------------------- observables

template <typename T>
using Observable = std::function<double(const Model<T>&)>;

inline std::vector<std::string> default_observables(const ModelSpec& m, const InitSpec& init, const std::string& task) {
    if (is_continuous(m.name))
        return task == "stationary" ? std::vector<std::string>{"range"}
                : task == "ensemble" ? std::vector<std::string>{"clusters", "range"}
                : std::vector<std::string>{"mean", "range", "clusters"};
    if (m.name == "potts")
        return task == "stationary" ? std::vector<std::string>{"order"} : std::vector<std::string>{"order", "energy"};
    if (init.multi)
        return task == "stationary" ? std::vector<std::string>{"largest"} : std::vector<std::string>{"opinions", "largest", "rho"};
    if (m.name == "ising")
        return task == "stationary" ? std::vector<std::string>{"m"} : std::vector<std::string>{"m", "energy"};

    return task == "stationary" ? std::vector<std::string>{"m"} : std::vector<std::string>{"m", "rho"};
}

// per_interval: for "activity", measure since the previous evaluation of this
// observable (time series) instead of since the start (final values). The
// returned function then carries state, so use one per trajectory.
template <typename T>
Observable<T> make_observable(const std::string& name, const ModelSpec& m, bool per_interval = false) {
    const bool continuous = is_continuous(m.name);
    auto need = [&](bool ok, const std::string& what) {
        if (!ok)
            throw OptionError("observable '" + name + "' " + what + " (model " + m.name + ")");
    };
    if (name == "m" || name == "mean")
        return [](const Model<T>& x) { return x.get_magnetization_per_site(); };
    if (name == "abs_m")
        return [](const Model<T>& x) { return std::fabs(x.get_magnetization_per_site()); };
    if (name == "omega")
        return [](const Model<T>& x) { return x.get_degree_weighted_magnetization(); };
    // Sweeps from the update counter, so it is right whether the run advanced by sweeps or by single updates.
    if (name == "sweeps") return [](const Model<T>& x) {
        return static_cast<double>(x.get_updates()) / static_cast<double>(x.get_graph().size());
    };
    if (name == "rho") {
        need(!continuous, "is for discrete opinions");
        return [](const Model<T>& x) { return x.get_active_link_density(); };
    }
    if (name == "persistence")
        return [](const Model<T>& x) { return x.persistence(); };
    if (name == "variance")
        return [](const Model<T>& x) { return x.opinion_variance(); };
    if (name == "bimodality")
        return [](const Model<T>& x) { return x.bimodality_coefficient(); };
    if (name == "activity") {
        if (!per_interval)
            return [](const Model<T>& x) {
                return x.get_updates() == 0 ? 0.0
                     : static_cast<double>(x.get_changes()) / static_cast<double>(x.get_updates());
            };
        auto last = std::make_shared<std::pair<long long, long long>>(0, 0);   // (updates, changes) at the previous reading
        return [last](const Model<T>& x) {
            const long long du = x.get_updates() - last->first, dc = x.get_changes() - last->second;
            *last = {x.get_updates(), x.get_changes()};
            return du == 0 ? 0.0 : static_cast<double>(dc) / static_cast<double>(du);
        };
    }
    if constexpr (std::is_integral<T>::value) {
        if (name == "opinions")
            return [](const Model<T>& x) { return static_cast<double>(x.number_of_opinions()); };
        if (name == "largest")
            return [](const Model<T>& x) { return x.largest_opinion_fraction(); };
        if (name == "entropy")
            return [](const Model<T>& x) { return x.opinion_entropy(); };
        if (name == "effective_opinions")
            return [](const Model<T>& x) { return x.effective_number_of_opinions(); };
        if (name == "segregation")
            return [](const Model<T>& x) { return x.segregation(); };
        if (name == "energy") {
            need(m.name == "ising" || m.name == "potts", "needs the ising or potts model");
            return [](const Model<T>& x) {
                if (const auto* i = dynamic_cast<const IsingModel<T>*>(&x))
                    return i->energy_per_site();
                return static_cast<const PottsModel<T>&>(x).energy_per_site();
            };
        }
        if (name == "order") {
            need(m.name == "potts", "needs the potts model");
            return [](const Model<T>& x) { return static_cast<const PottsModel<T>&>(x).order_parameter(); };
        }
        need(name != "range" && name != "clusters" && name != "largest_cluster", "is for continuous opinions");
    }
    else {
        if (name == "range")
            return [](const Model<T>& x) { return static_cast<const BoundedConfidenceModel<T>&>(x).opinion_range(); };
        if (name == "clusters")
            return [](const Model<T>& x) { return static_cast<double>(count_clusters(x.get_graph().states(), 1e-3, 0.01)); };
        if (name == "largest_cluster")
            return [](const Model<T>& x) { return largest_cluster_fraction(x.get_graph().states(), 1e-3); };
        need(name != "opinions" && name != "largest" && name != "energy" && name != "order" && name != "entropy" &&
             name != "effective_opinions" && name != "segregation", "is for discrete opinions");
    }
    std::vector<std::string> names;
    for (const auto& d : observable_docs())
        names.push_back(d.name);
    const std::string s = suggestion(name, names);
    throw OptionError("unknown observable '" + name + "'" + (s.empty() ? "" : " (did you mean '" + s + "'?)") +
                      "; see --help observables");
}

// ---- the factory ---- //

template <typename T>
struct Setup {
    ModelSpec    model;
    TopologySpec topology_spec;
    std::shared_ptr<const Topology> topology;   // the shared network (when not regenerated)
    InitSpec     init;
    bool         regenerate = false;
    typename Ensemble<T>::ModelFactory factory;
    std::string  description;
};

// graph_seed fixes the shared random network, so it is the same in every
// task and at every point of a sweep
// Pass a cache to share built networks between setups (sweeps).
template <typename T>
Setup<T> read_setup(Options& o, unsigned graph_seed, TopologyCache* cache = nullptr) {
    Setup<T> s;
    s.model = read_model(o);
    s.topology_spec = read_topology(o);
    s.init = read_init(o, s.model);
    s.regenerate = s.topology_spec.random ? o.flag("regenerate-graph", false) : false;
    TopologyCache local;
    s.topology = (cache ? cache : &local)->get(s.topology_spec, graph_seed);
    if (s.topology->nodes == 0)
        throw OptionError("the graph has no nodes");

    const ModelSpec model = s.model;
    const TopologySpec spec = s.topology_spec;
    const InitSpec init = s.init;
    const bool regenerate = s.regenerate;
    const std::shared_ptr<const Topology> shared = s.topology;
    s.factory = [model, spec, init, regenerate, shared](std::mt19937& rng) -> std::unique_ptr<Model<T>> {
        std::shared_ptr<const Topology> topo = shared;
        if (regenerate)
            topo = std::make_shared<const Topology>(build_topology(spec, rng));
        auto g = make_graph<T>(*topo, make_states<T>(init, topo->nodes, rng));
        const std::vector<std::size_t> zealots = place_zealots(init, *g, rng);
        auto m = make_model<T>(model, std::move(g), &rng);
        if (!zealots.empty())
            m->set_frozen(zealots);
        return m;
    };
    s.description = s.model.description + " on " + s.topology->description + (s.regenerate ? " (new network per replica)" : "") +
                    "; start: " + s.init.description;
    return s;
}

} // namespace driver

#endif // ODSP_DRIVER_SETUP_HPP
