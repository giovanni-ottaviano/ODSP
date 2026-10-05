#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>
#include <algorithm>

#include "BaseGraph.hpp"
#include "RegularLattice.hpp"
#include "CompleteGraph.hpp"
#include "VoterModel.hpp"
#include "MajorityModel.hpp"
#include "Ensemble.hpp"
#include "Utils.hpp"
#include "Cli.hpp"
#include "CsvWriter.hpp"

// Compact text histogram for the consensus-time distribution.
static void histogram(const std::vector<long long>& data, int bins = 10, int width = 40) {
    if (data.empty()) { std::cout << "  (no data)\n"; return; }
    const long long lo = *std::min_element(data.begin(), data.end());
    const long long hi = *std::max_element(data.begin(), data.end());
    const double span = std::max<double>(1.0, static_cast<double>(hi - lo));
    std::vector<int> count(bins, 0);
    for (long long v : data) {
        int b = static_cast<int>((v - lo) / span * (bins - 1) + 0.5);
        count[std::min(std::max(b, 0), bins - 1)]++;
    }
    const int peak = *std::max_element(count.begin(), count.end());
    for (int b = 0; b < bins; ++b) {
        const long long edge = lo + static_cast<long long>(span * b / (bins - 1) + 0.5);
        const int bar = peak ? count[b] * width / peak : 0;
        std::cout << "  " << std::setw(7) << edge << " | "
                  << std::string(bar, '#') << " " << count[b] << '\n';
    }
}

int main(int argc, char** argv) {
    const unsigned default_seed = 2024;
    const CliOptions opt = parse_cli(argc, argv,
        "Ensemble statistics: consensus times, exit probability, majority freezing.",
        &default_seed);
    std::mt19937 seeder(opt.seed);   // one ensemble seed per section

    // ---- 1) Consensus-time distribution: voter model on a 16x16 lattice ----
    {
        const std::vector<int> dims = {16, 16};
        const std::size_t N = 16 * 16;
        auto factory = [dims, N](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            auto g = std::make_unique<RegularLattice<int>>(create_random_lattice(N, 0.5, rng), dims);
            return std::make_unique<VoterModel<int>>(std::move(g), &rng);
        };
        Ensemble<int> ens(factory, seeder());
        auto res = ens.run(/*replicas=*/300, /*max_steps=*/40000, true, opt.threads);

        std::cout << "=== Voter / 16x16 lattice: consensus-time distribution ("
                  << res.replicas << " replicas) ===\n";
        std::cout << "reached consensus : " << res.consensus_count << " / " << res.replicas
                  << "   (survival " << std::fixed << std::setprecision(3)
                  << res.survival_fraction << ")\n";
        std::cout << "consensus time    : mean " << std::setprecision(1) << res.mean_consensus_time
                  << " sweeps, stddev " << res.stddev_consensus_time
                  << ", median " << res.median_consensus_time << "\n";
        if (res.survival_fraction > 0.0)
            std::cout << "                    (censored runs: true mean >= "
                      << res.mean_consensus_time_lower_bound << ")\n";
        std::cout << "exit prob (+1)    : " << std::setprecision(3) << res.exit_prob_up
                  << "   (expected ~0.5 from f0=0.5)\n";
        std::cout << "distribution (sweeps):\n";
        histogram(res.consensus_times);
        std::cout << '\n';

        CsvWriter csv(opt.csv_path("consensus_times"), {"replica", "consensus_sweeps"});
        for (std::size_t i = 0; i < res.consensus_times.size(); ++i)
            csv.row(i, res.consensus_times[i]);
    }

    // ---- 2) Exit-probability law: voter is conservative, so P(+1) = f0 ----
    {
        const std::size_t N = 200;
        std::cout << "=== Voter / complete graph: exit probability vs initial up-fraction ===\n";
        std::cout << "   f0     P(+1)    (voter on a regular graph: P(+1) = f0 exactly)\n";
        CsvWriter csv(opt.csv_path("exit_prob"),
                      {"f0", "exit_prob_up", "consensus_count", "replicas", "mean_consensus_sweeps"});
        for (double f0 : {0.30, 0.45, 0.50, 0.65, 0.80}) {
            auto factory = [N, f0](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
                auto g = std::make_unique<CompleteGraph<int>>(create_random_lattice(N, f0, rng));
                return std::make_unique<VoterModel<int>>(std::move(g), &rng);
            };
            Ensemble<int> ens(factory, seeder());
            auto res = ens.run(/*replicas=*/500, /*max_steps=*/5000, true, opt.threads);
            std::cout << "  " << std::fixed << std::setprecision(2) << std::setw(5) << f0
                      << "   " << std::setprecision(3) << std::setw(6) << res.exit_prob_up << '\n';
            csv.row(f0, res.exit_prob_up, res.consensus_count, res.replicas, res.mean_consensus_time);
        }
        std::cout << '\n';
    }

    // ---- 3) Majority on a lattice: how often does it FREEZE vs order? ----
    {
        const std::vector<int> dims = {32, 32};
        const std::size_t N = 32 * 32;
        auto factory = [dims, N](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            auto g = std::make_unique<RegularLattice<int>>(create_random_lattice(N, 0.5, rng), dims);
            return std::make_unique<MajorityModel<int>>(std::move(g), &rng);
        };
        Ensemble<int> ens(factory, seeder());
        auto res = ens.run(/*replicas=*/200, /*max_steps=*/3000, true, opt.threads);

        std::cout << "=== Majority / 32x32 lattice: freezing statistics ("
                  << res.replicas << " replicas) ===\n";
        std::cout << "reached consensus : " << res.consensus_count << " / " << res.replicas << "\n";
        std::cout << "froze (survival)  : " << std::setprecision(3) << res.survival_fraction << "\n";
        std::cout << "mean rho at stop  : " << res.mean_final_rho
                  << "   (>0 => persistent domain walls in frozen runs)\n";
        std::cout << "Binder cumulant   : " << res.binder
                  << ",  susceptibility " << std::setprecision(2) << res.susceptibility << "\n";

        CsvWriter csv(opt.csv_path("majority"),
                      {"replicas", "consensus_count", "survival_fraction", "mean_final_rho",
                       "m1", "m2", "m4", "susceptibility", "binder"});
        csv.row(res.replicas, res.consensus_count, res.survival_fraction, res.mean_final_rho,
                res.m1, res.m2, res.m4, res.susceptibility, res.binder);
    }

    if (opt.csv_enabled())
        std::cout << "\nCSV written: " << opt.csv_path("consensus_times") << ", "
                  << opt.csv_path("exit_prob") << ", " << opt.csv_path("majority") << '\n';
    return 0;
}
