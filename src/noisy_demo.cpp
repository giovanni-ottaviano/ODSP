#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>

#include "BaseGraph.hpp"
#include "CompleteGraph.hpp"
#include "NoisyVoterModel.hpp"
#include "Utils.hpp"
#include "Cli.hpp"
#include "CsvWriter.hpp"

// Measure the stationary order parameter <|m|>, susceptibility and Binder
// cumulant (with jackknife errors) for the noisy voter on a complete graph
// (mean field) of N nodes, as a function of the spontaneous-flip rate `noise`.
// tau is the autocorrelation time of |m| in samples: the errors are reliable
// when it is much smaller than the jackknife block (4000/20 = 200 samples).
static void sweep(std::size_t N, std::mt19937& rng, CsvWriter& csv) {
    std::cout << "N = " << N << "\n";
    std::cout << "   noise        <|m|>                chi'               Binder         tau\n";
    for (double noise : {0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}) {
        auto g = std::make_unique<CompleteGraph<int>>(create_random_lattice(N, 0.5, rng));
        NoisyVoterModel<int> m(std::move(g), &rng, noise);
        auto s = m.sample_stationary(/*burn_in=*/2000, /*n_samples=*/4000, /*interval=*/2);
        std::cout << "  " << std::fixed
                  << std::setprecision(3) << std::setw(6) << noise << "   "
                  << std::setprecision(3) << std::setw(6) << s.abs_mean << " +- " << std::setw(5) << s.abs_mean_err << "   "
                  << std::setprecision(2) << std::setw(7) << s.abs_susceptibility << " +- "
                  << std::setw(5) << s.abs_susceptibility_err << "   "
                  << std::setprecision(3) << std::setw(6) << s.binder << " +- " << std::setw(5) << s.binder_err << "   "
                  << std::setprecision(1) << std::setw(5) << s.tau_abs_m << '\n';
        csv.row(N, noise, s.mean, s.mean_err, s.abs_mean, s.abs_mean_err, s.m2, s.m4,
                s.susceptibility, s.susceptibility_err, s.abs_susceptibility, s.abs_susceptibility_err,
                s.binder, s.binder_err, s.tau_m, s.tau_abs_m);
    }
    std::cout << '\n';
}

int main(int argc, char** argv) {
    const CliOptions opt = parse_cli(argc, argv,
        "Noisy voter on a complete graph: stationary order parameter vs noise.");
    std::mt19937 rng(opt.seed);
    CsvWriter csv(opt.csv_path("sweep"),
                  {"N", "noise", "m_mean", "m_mean_err", "abs_m_mean", "abs_m_mean_err", "m2", "m4",
                   "susceptibility", "susceptibility_err", "abs_susceptibility", "abs_susceptibility_err",
                   "binder", "binder_err", "tau_m", "tau_abs_m"});

    std::cout << "=== Noisy voter on a complete graph: noise-driven order/disorder ===\n";
    std::cout << "(low noise: ordered, <|m|>~1, Binder~2/3 ; high noise: disordered, <|m|>~0, Binder~0)\n";
    std::cout << "(chi' = N (<m^2> - <|m|>^2); errors are block-jackknife standard errors)\n\n";

    sweep(64, rng, csv);
    sweep(256, rng, csv);

    std::cout << "The crossover noise shifts down as N grows (mean-field noisy voter:\n"
                 "the ordered window scales like ~1/N), so the two Binder curves cross\n"
                 "at different noise -- exactly the finite-size behaviour to exploit.\n";
    if (csv.enabled()) std::cout << "\nCSV written: " << csv.path() << '\n';
    return 0;
}
