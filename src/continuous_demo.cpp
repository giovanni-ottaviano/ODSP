#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "AdjacencyGraph.hpp"
#include "CompleteGraph.hpp"
#include "Generators.hpp"
#include "RegularLattice.hpp"
#include "Ensemble.hpp"
#include "OpinionClusters.hpp"
#include "DeffuantModel.hpp"
#include "HegselmannKrauseModel.hpp"
#include "Utils.hpp"
#include "Cli.hpp"
#include "CsvWriter.hpp"

// Continuous opinions with bounded confidence: agents only listen to opinions
// within epsilon of their own.
//   1) number of final opinion clusters vs epsilon, Deffuant-Weisbuch vs
//      Hegselmann-Krause on a complete graph, against the 1/(2 epsilon) rule;
//   2) what a fragmented final state looks like;
//   3) the effect of the network: complete graph vs lattice vs scale-free.

namespace {

using Factory = Ensemble<double>::ModelFactory;

double major_clusters(const Model<double>& m) {
    return static_cast<double>(count_clusters(m.get_graph().states(), 1e-3, 0.01));
}

double all_clusters(const Model<double>& m) {
    return static_cast<double>(count_clusters(m.get_graph().states(), 1e-3));
}

// Sweeps until 99% of the agents sit in settled clusters (width <= 1e-3), or
// until convergence if that comes first. Full convergence can take far longer:
// a few stragglers stranded near the extremes must still meet each other, and
// on a complete graph a given pair interacts only every ~N/2 sweeps.
double settle_time(Model<double>& m, long long cap) {
    const double n = static_cast<double>(m.get_graph().size());
    for (long long s = 0; s < cap; ++s) {
        if (m.advance(1, true, true), m.is_absorbing_state()) break;
        std::size_t settled = 0;
        for (const OpinionCluster& c : opinion_clusters(m.get_graph().states(), 1e-3))
            if (c.max - c.min <= 1e-3) settled += c.size;
        if (static_cast<double>(settled) >= 0.99 * n) break;
    }
    return static_cast<double>(m.get_mc_steps());
}

} // namespace

int main(int argc, char** argv) {
    const CliOptions opt = parse_cli(argc, argv,
        "Bounded-confidence models (Deffuant-Weisbuch, Hegselmann-Krause): clusters vs confidence and network.");
    std::mt19937 seeder(opt.seed);
    std::cout << std::fixed;
    const std::size_t N = 1000;
    const int replicas = 24;

    // ---- 1) Clusters vs epsilon ----
    {
        std::cout << "=== 1) Final opinion clusters vs confidence bound, complete graph N = " << N
                  << ", opinions uniform in [0,1], " << replicas << " runs ===\n"
                  << "(major cluster: >= 1% of the agents; DW settle: sweeps until 99% of the agents\n"
                  << " are in settled clusters; HK rounds: rounds to full convergence)\n"
                  << "  epsilon  1/(2eps)   DW clusters   DW settle   HK clusters   HK rounds\n";
        CsvWriter csv(opt.csv_path("clusters"), {"epsilon", "dw_major_clusters", "dw_major_clusters_sd",
                                                 "dw_settle_sweeps",
                                                 "hk_major_clusters", "hk_major_clusters_sd", "hk_rounds"});
        for (double eps : {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4}) {
            const Factory dw = [eps](std::mt19937& rng) -> std::unique_ptr<Model<double>> {
                return std::make_unique<DeffuantModel<>>(std::make_unique<CompleteGraph<double>>(create_uniform_opinions(N, rng)),
                                                         &rng, eps);
            };
            const Factory hk = [eps](std::mt19937& rng) -> std::unique_ptr<Model<double>> {
                return std::make_unique<HegselmannKrauseModel<>>(
                    std::make_unique<CompleteGraph<double>>(create_uniform_opinions(N, rng)), &rng, eps);
            };
            const unsigned seed_dw = seeder(), seed_hk = seeder();
            // Same seeds twice: the time run reproduces the cluster run exactly.
            const auto dw_c = Ensemble<double>(dw, seed_dw).run(replicas, 10000000, true, opt.threads, major_clusters);
            // Settling time: the same replicas, stepped sweep by sweep (factory
            // with a fixed seed per replica, as Ensemble does).
            double dw_settle = 0.0;
            {
                std::mt19937 seeds(seed_dw);
                for (int r = 0; r < replicas; ++r) {
                    std::mt19937 rng(seeds());
                    auto m = dw(rng);
                    dw_settle += settle_time(*m, 100000) / replicas;
                }
            }
            const auto hk_c = Ensemble<double>(hk, seed_hk).run(replicas, 100000, true, opt.threads, major_clusters);
            const auto hk_t = Ensemble<double>(hk, seed_hk).run(replicas, 100000, true, opt.threads,
                [](const Model<double>& m) { return static_cast<double>(m.get_mc_steps()); });

            std::cout << "  " << std::setprecision(2) << std::setw(5) << eps << "    " << std::setprecision(1)
                      << std::setw(5) << 1.0 / (2.0 * eps) << "     " << std::setw(4) << dw_c.mean_final_value
                      << " +- " << std::setprecision(1) << dw_c.stddev_final_value << "   " << std::setprecision(0)
                      << std::setw(6) << dw_settle << "      " << std::setprecision(1) << std::setw(4)
                      << hk_c.mean_final_value << " +- " << hk_c.stddev_final_value << "   " << std::setprecision(1)
                      << std::setw(7) << hk_t.mean_final_value << '\n';
            csv.row(eps, dw_c.mean_final_value, dw_c.stddev_final_value, dw_settle,
                    hk_c.mean_final_value, hk_c.stddev_final_value, hk_t.mean_final_value);
        }
        std::cout << "  (both models fragment into ~1/(2 eps) clusters; consensus above eps ~ 0.27 (DW)\n"
                     "   and ~0.2 (HK). DW settles in a few dozen sweeps; FULL convergence often takes\n"
                     "   thousands, waiting for 2-4 stragglers near the extremes to meet each other)\n\n";
    }

    // ---- 2) A fragmented final state ----
    {
        const double eps = 0.15;
        std::mt19937 rng(seeder());
        DeffuantModel<> m(std::make_unique<CompleteGraph<double>>(create_uniform_opinions(N, rng)), &rng, eps);
        m.advance(10000000);
        std::cout << "=== 2) One Deffuant run, epsilon = " << std::setprecision(2) << eps << ", converged after " << m.get_mc_steps()
                  << " sweeps ===\n  opinion    agents\n";
        CsvWriter csv(opt.csv_path("final_state"), {"position", "size", "min", "max"});
        for (const OpinionCluster& c : m.clusters()) {
            std::cout << "  " << std::setprecision(4) << c.position << "   " << std::setw(5) << c.size << "  "
                      << std::string(c.size / 10, '#') << '\n';
            csv.row(c.position, c.size, c.min, c.max);
        }
        std::cout << "  (major clusters about 2 eps apart, plus a few stragglers near the extremes)\n\n";
    }

    // ---- 3) Network effect ----
    {
        const double eps = 0.25;
        std::cout << "=== 3) Deffuant on different networks, N ~ 1000, epsilon = " << std::setprecision(2) << eps << ", " << replicas << " runs ===\n"
                  << "  network               major clusters   all clusters   sweeps\n";
        CsvWriter csv(opt.csv_path("networks"), {"network", "major_clusters", "all_clusters", "sweeps"});
        using Graph = std::unique_ptr<BaseGraph<double>>;
        const std::vector<std::pair<const char*, std::function<Graph(std::mt19937&)>>> nets = {
            {"complete graph", [](std::mt19937& r) -> Graph {
                return std::make_unique<CompleteGraph<double>>(create_uniform_opinions(N, r)); }},
            {"Barabasi-Albert m=3", [](std::mt19937& r) -> Graph {
                const EdgeList e = barabasi_albert_edges(N, 3, r);
                return std::make_unique<AdjacencyGraph<double>>(create_uniform_opinions(N, r), e); }},
            {"square lattice 32x32", [](std::mt19937& r) -> Graph {
                return std::make_unique<RegularLattice<double>>(create_uniform_opinions(1024, r), std::vector<int>{32, 32}); }},
        };
        for (const auto& [name, make] : nets) {
            const Factory f = [make = make, eps](std::mt19937& rng) -> std::unique_ptr<Model<double>> {
                return std::make_unique<DeffuantModel<>>(make(rng), &rng, eps);
            };
            const unsigned seed = seeder();
            const auto major = Ensemble<double>(f, seed).run(replicas, 10000000, true, opt.threads, major_clusters);
            const auto all   = Ensemble<double>(f, seed).run(replicas, 10000000, true, opt.threads, all_clusters);
            const auto time  = Ensemble<double>(f, seed).run(replicas, 10000000, true, opt.threads,
                [](const Model<double>& m) { return static_cast<double>(m.get_mc_steps()); });
            std::cout << "  " << std::left << std::setw(22) << name << std::right << std::setprecision(1)
                      << std::setw(8) << major.mean_final_value << std::setw(16) << all.mean_final_value
                      << std::setprecision(0) << std::setw(11) << time.mean_final_value << '\n';
            csv.row(name, major.mean_final_value, all.mean_final_value, time.mean_final_value);
        }
        std::cout << "  (sparse networks leave many small clusters of isolated agents who disagree\n"
                     "   with all their neighbours, and converge more slowly)\n";
    }
    return 0;
}
