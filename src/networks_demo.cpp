#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "AdjacencyGraph.hpp"
#include "EdgeList.hpp"
#include "EdgeListIO.hpp"
#include "Generators.hpp"
#include "VoterModel.hpp"
#include "Ensemble.hpp"
#include "Utils.hpp"
#include "Cli.hpp"
#include "CsvWriter.hpp"

// The voter model on complex networks: how topology shapes the dynamics.
//
// 1) Same size and mean degree, different structure: active-link plateau and
//    consensus time on random-regular, Erdos-Renyi, Barabasi-Albert and
//    Watts-Strogatz networks. On uncorrelated random networks the pair
//    approximation predicts rho = rho* (1 - w^2), with w the degree-weighted
//    magnetization and rho* = (mu-2) / (2(mu-1)), mu the mean degree (Vazquez &
//    Eguiluz, New J. Phys. 10, 063011, 2008). In a finite system w drifts away
//    from 0 during a run, so rho/(1 - w^2) is the size-independent estimate of
//    rho*. Clustered small worlds deviate from the prediction.
// 2) Which magnetization is conserved: exit probability on a scale-free network
//    started with the hubs up, under the three voter update schemes.
// 3) Optionally, the same analysis on a network read from an edge-list file.

namespace {

struct Network {
    std::string name;
    std::size_t nodes;
    EdgeList    edges;
};

// Mean of rho / (1 - w^2) over sweeps (from, to], over a few independent runs.
double plateau_rho(const Network& net, std::mt19937& rng, long long from, long long to, int runs) {
    double sum = 0.0;
    long long count = 0;
    for (int r = 0; r < runs; ++r) {
        VoterModel<int> m(std::make_unique<AdjacencyGraph<int>>(create_random_lattice(net.nodes, 0.5, rng), net.edges),
                          &rng);
        m.advance(to, true, /*stop_when_absorbed=*/false,
                  [&](const Model<int>& model) {
                      if (model.get_mc_steps() <= from) return;
                      const double w = model.get_degree_weighted_magnetization();
                      if (1.0 - w * w < 1e-12) return;   // at consensus rho = 0 and 1 - w^2 = 0: no information
                      sum += model.get_active_link_density() / (1.0 - w * w);
                      ++count;
                  });
    }
    return count ? sum / static_cast<double>(count) : std::numeric_limits<double>::quiet_NaN();   // NaN: no sample
}

void analyse(const Network& net, std::mt19937& seeder, int threads, CsvWriter& csv) {
    const auto deg = degree_sequence(net.nodes, net.edges);
    double k1 = 0.0, k2 = 0.0;
    for (std::size_t k : deg) { k1 += static_cast<double>(k); k2 += static_cast<double>(k) * k; }
    k1 /= static_cast<double>(net.nodes);
    k2 /= static_cast<double>(net.nodes);
    const std::size_t kmax = *std::max_element(deg.begin(), deg.end());
    const double predicted = (k1 - 2.0) / (2.0 * (k1 - 1.0));

    std::mt19937 rng(seeder());
    const double rho = plateau_rho(net, rng, /*from=*/20, /*to=*/100, /*runs=*/4);

    const long long cap = 10000;
    const Network* p = &net;
    Ensemble<int> ens([p](std::mt19937& r) -> std::unique_ptr<Model<int>> {
        return std::make_unique<VoterModel<int>>(
            std::make_unique<AdjacencyGraph<int>>(create_random_lattice(p->nodes, 0.5, r), p->edges), &r);
    }, seeder());
    const auto res = ens.run(/*replicas=*/96, cap, true, threads);

    std::cout << "  " << std::left << std::setw(22) << net.name << std::right << std::fixed
              << std::setw(6) << net.nodes << std::setw(7) << net.edges.size()
              << std::setprecision(2) << std::setw(7) << k1 << std::setw(6) << kmax
              << std::setw(8) << k2 / (k1 * k1)
              << std::setprecision(3) << std::setw(8);
    if (std::isnan(rho)) std::cout << "n/a";   // every run reached consensus before sampling began
    else                 std::cout << rho;
    std::cout << std::setw(8) << predicted;
    if (res.consensus_count == res.replicas)
        std::cout << std::setprecision(0) << std::setw(9) << res.mean_consensus_time
                  << std::setw(9) << res.median_consensus_time << '\n';
    else
        std::cout << "   " << res.replicas - res.consensus_count << "/" << res.replicas
                  << " not done by " << cap << " (median "
                  << (std::isnan(res.median_consensus_time) ? std::string("> cap")
                                                             : std::to_string(static_cast<long long>(res.median_consensus_time)))
                  << ")\n";

    csv.row(net.name, net.nodes, net.edges.size(), k1, kmax, k2 / (k1 * k1), rho, predicted,
            res.consensus_count, res.replicas, res.mean_consensus_time, res.median_consensus_time,
            res.mean_consensus_time_lower_bound);
}

void print_header() {
    std::cout << "  network                 nodes  edges    <k>  kmax  <k2>/<k>2   rho*  pair-ap   T_mean  T_median\n";
}

} // namespace

int main(int argc, char** argv) {
    const CliOptions opt = parse_cli(argc, argv,
        "Voter model on complex networks: active-link plateau, consensus time, exit probability.",
        nullptr, {{"--edges", "<file>", "also analyse the network in this edge-list file"}});
    std::mt19937 seeder(opt.seed);
    std::mt19937 build(seeder());

    // Read the optional input network first, so a bad path fails immediately.
    const std::string path = opt.get("--edges");
    EdgeListFile f;
    if (!path.empty()) {
        try {
            f = read_edge_list(path);
        } catch (const std::exception& e) {
            std::cerr << argv[0] << ": " << e.what() << '\n';
            return 1;
        }
    }

    CsvWriter topo_csv(opt.csv_path("topologies"),
                       {"network", "nodes", "edges", "mean_degree", "max_degree", "degree_heterogeneity",
                        "rho_plateau", "rho_pair_approx", "consensus_count", "replicas",
                        "mean_consensus_sweeps", "median_consensus_sweeps", "mean_consensus_lower_bound"});

    // ---- 1) Topology comparison at N = 1000, <k> = 6 ----
    const std::size_t N = 1000;
    std::vector<Network> nets;
    nets.push_back({"random regular k=6", N, random_regular_edges(N, 6, build)});
    {
        const Subgraph lcc = largest_connected_component(N, gnm_edges(N, 3 * N, build));
        nets.push_back({"Erdos-Renyi (LCC)", lcc.nodes, lcc.edges});
    }
    nets.push_back({"Barabasi-Albert m=3", N, barabasi_albert_edges(N, 3, build)});
    nets.push_back({"Watts-Strogatz b=1", N, watts_strogatz_edges(N, 6, 1.0, build)});
    nets.push_back({"Watts-Strogatz b=0.1", N, watts_strogatz_edges(N, 6, 0.1, build)});
    nets.push_back({"Watts-Strogatz b=0.01", N, watts_strogatz_edges(N, 6, 0.01, build)});

    std::cout << "=== Voter model on networks with N = 1000, <k> = 6 ===\n"
              << "(rho* = rho/(1-w^2) averaged over sweeps 20-100, w = degree-weighted magnetization;\n"
              << " pair-ap = (<k>-2)/(2(<k>-1)), accurate on uncorrelated random graphs, not on clustered\n"
              << " ones; T = consensus time in sweeps over 96 runs)\n\n";
    print_header();
    for (const Network& net : nets) analyse(net, seeder, opt.threads, topo_csv);
    std::cout << "\n  Heterogeneous degrees (large <k2>/<k>2) speed up consensus (T ~ N <k>2/<k2>);\n"
                 "  clustering (small beta) lowers rho* below the pair approximation and slows\n"
                 "  consensus toward the 1D ring's T ~ N^2.\n\n";

    // ---- 2) Exit probability on a scale-free network, hubs up ----
    {
        const std::size_t n = 1000;
        const EdgeList edges = barabasi_albert_edges(n, 3, build);
        const auto deg = degree_sequence(n, edges);
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&deg](std::size_t a, std::size_t b) { return deg[a] > deg[b]; });
        std::vector<int> states(n, -1);
        for (std::size_t r = 0; r < n / 10; ++r) states[order[r]] = 1;

        double sk = 0, sks = 0, ss = 0, sinv = 0, sinvs = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double k = static_cast<double>(deg[i]);
            sk += k; sks += k * states[i]; ss += states[i]; sinv += 1.0 / k; sinvs += states[i] / k;
        }
        const double omega = sks / sk, m = ss / n, inv = sinvs / sinv;

        std::cout << "=== Exit probability, Barabasi-Albert N = 1000, top 10% of nodes by degree up ===\n"
                  << "(each scheme conserves a different weighted magnetization M; P(+1) = (1 + M0)/2)\n\n"
                  << "  update    conserved M                M0       predicted  measured\n";
        CsvWriter exit_csv(opt.csv_path("exit"), {"update", "M0", "predicted", "measured", "replicas"});
        const struct { VoterUpdate u; const char* name; const char* what; double M0; } schemes[] = {
            {VoterUpdate::Node,    "node",    "sum k s / sum k        ", omega},
            {VoterUpdate::Link,    "link",    "sum s / N              ", m},
            {VoterUpdate::Reverse, "reverse", "sum (s/k) / sum (1/k)  ", inv},
        };
        for (const auto& s : schemes) {
            Ensemble<int> ens([&edges, &states, u = s.u](std::mt19937& r) -> std::unique_ptr<Model<int>> {
                return std::make_unique<VoterModel<int>>(std::make_unique<AdjacencyGraph<int>>(states, edges), &r, u);
            }, seeder());
            const auto res = ens.run(/*replicas=*/1000, 10000000, true, opt.threads);
            const double predicted = (1.0 + s.M0) / 2.0;
            std::cout << "  " << std::left << std::setw(9) << s.name << std::right << s.what << std::fixed
                      << std::setprecision(3) << std::setw(7) << s.M0 << std::setw(12) << predicted
                      << std::setw(10) << res.exit_prob_up << '\n';
            exit_csv.row(s.name, s.M0, predicted, res.exit_prob_up, res.replicas);
        }
        std::cout << "  (1000 runs each: statistical error ~0.015)\n\n";
    }

    // ---- 3) A network from file ----
    if (!path.empty()) {
        const Subgraph lcc = largest_connected_component(f.nodes, f.edges);
        std::cout << "=== Network from " << path << " ===\n"
                  << "  read " << f.nodes << " nodes, " << f.edges.size() << " edges (dropped "
                  << f.self_loops_dropped << " self-loops, " << f.duplicates_dropped << " repeated edges);\n"
                  << "  largest connected component: " << lcc.nodes << " nodes, " << lcc.edges.size() << " edges\n\n";
        print_header();
        analyse({"from file (LCC)", lcc.nodes, lcc.edges}, seeder, opt.threads, topo_csv);
        std::cout << '\n';
    }

    if (opt.csv_enabled())
        std::cout << "CSV written: " << topo_csv.path() << ", " << opt.csv_path("exit") << '\n';
    return 0;
}
