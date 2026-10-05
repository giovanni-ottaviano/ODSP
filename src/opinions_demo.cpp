#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "CompleteGraph.hpp"
#include "RegularLattice.hpp"
#include "Ensemble.hpp"
#include "VoterModel.hpp"
#include "MajorityModel.hpp"
#include "PottsModel.hpp"
#include "Utils.hpp"
#include "Cli.hpp"
#include "CsvWriter.hpp"

// Models with more than two opinions:
//   1) voter model with q opinions: who wins (P = initial share) and how long
//      it takes from N distinct opinions (exactly (N-1)^2 updates on a
//      complete graph);
//   2) how the number of surviving opinions decays in time: 2D lattice vs
//      complete graph, voter vs plurality (majority) rule;
//   3) q-state Potts model on the square lattice across T_c = 1/ln(1 + sqrt q):
//      continuous for q = 3, first order (energy jump) for q = 8.

int main(int argc, char** argv) {
    const CliOptions opt = parse_cli(argc, argv,
        "Multi-opinion dynamics: q-opinion voter, opinion survival, Potts transitions.");
    std::mt19937 seeder(opt.seed);
    std::cout << std::fixed;

    // ---- 1) q-opinion voter: fixation and consensus time ----
    {
        const std::size_t N = 100;
        const std::vector<double> shares = {0.1, 0.2, 0.3, 0.4};
        Ensemble<int> ens([N, &shares](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            return std::make_unique<VoterModel<int>>(
                std::make_unique<CompleteGraph<int>>(create_opinions_with_fractions(N, shares, rng)), &rng);
        }, seeder());
        const auto res = ens.run(2000, 10000000, true, opt.threads);
        std::cout << "=== 1) Voter model with 4 opinions, complete graph N = " << N << ", 2000 runs ===\n"
                  << "  opinion  initial share  P(wins)\n";
        CsvWriter csv(opt.csv_path("fixation"), {"opinion", "initial_share", "fixation_probability"});
        for (int k = 0; k < 4; ++k) {
            std::cout << "  " << std::setw(4) << k << "     " << std::setprecision(2) << std::setw(8) << shares[k]
                      << "       " << std::setprecision(3) << res.fixation_probability(k) << '\n';
            csv.row(k, shares[k], res.fixation_probability(k));
        }
        std::cout << "  (exact: P(wins) = initial share, for any number of opinions; error ~0.01)\n";

        Ensemble<int> distinct([N](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            return std::make_unique<VoterModel<int>>(std::make_unique<CompleteGraph<int>>(create_distinct_opinions(N)), &rng);
        }, seeder());
        const auto d = distinct.run(2000, 100000000LL, /*use_mc_steps=*/false, opt.threads);
        std::cout << "  From " << N << " distinct opinions: mean consensus time " << std::setprecision(0)
                  << d.mean_consensus_time << " updates +- " << d.stddev_consensus_time / std::sqrt(2000.0)
                  << "  (exact (N-1)^2 = " << (N - 1) * (N - 1) << ")\n\n";
    }

    // ---- 2) Surviving opinions over time ----
    {
        const int L = 64;
        const std::size_t N = static_cast<std::size_t>(L * L);
        std::cout << "=== 2) Number of surviving opinions from " << N << " distinct ones (average of 4 runs) ===\n"
                  << "  sweep   voter 2D   voter complete   plurality 2D\n";
        CsvWriter csv(opt.csv_path("survival"), {"sweep", "voter_2d", "voter_complete", "plurality_2d"});
        const std::vector<long long> times = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
        std::vector<std::vector<double>> curves(3, std::vector<double>(times.size(), 0.0));
        for (int run = 0; run < 4; ++run) {
            for (int which = 0; which < 3; ++which) {
                std::mt19937 rng(seeder());
                std::unique_ptr<Model<int>> m;
                if (which == 1)
                    m = std::make_unique<VoterModel<int>>(std::make_unique<CompleteGraph<int>>(create_distinct_opinions(N)), &rng);
                else {
                    auto g = std::make_unique<RegularLattice<int>>(create_distinct_opinions(N), std::vector<int>{L, L});
                    if (which == 0) m = std::make_unique<VoterModel<int>>(std::move(g), &rng);
                    else            m = std::make_unique<MajorityModel<int>>(std::move(g), &rng);
                }
                long long done = 0;
                for (std::size_t k = 0; k < times.size(); ++k) {
                    m->advance(times[k] - done, true, /*stop_when_absorbed=*/false);
                    done = times[k];
                    curves[which][k] += static_cast<double>(m->number_of_opinions()) / 4.0;
                }
            }
        }
        for (std::size_t k = 0; k < times.size(); ++k) {
            std::cout << "  " << std::setw(5) << times[k] << std::setprecision(1) << std::setw(11) << curves[0][k]
                      << std::setw(15) << curves[1][k] << std::setw(15) << curves[2][k] << '\n';
            csv.row(times[k], curves[0][k], curves[1][k], curves[2][k]);
        }
        std::cout << "  (complete graph: coalescence at the mean-field rate, (N-1)/t opinions; 2D voter:\n"
                     "   slower, scaling like N ln t / t; plurality rule freezes into many stable domains)\n\n";
    }

    // ---- 3) Potts transitions ----
    {
        const int L = 32;
        std::cout << "=== 3) q-state Potts model, " << L << "x" << L << " square lattice, heat bath, ordered start ===\n";
        CsvWriter csv(opt.csv_path("potts"), {"q", "T", "order", "order_err", "energy", "energy_err"});
        for (int q : {3, 8}) {
            const double Tc = 1.0 / std::log(1.0 + std::sqrt(static_cast<double>(q)));
            std::cout << "  q = " << q << ", T_c = " << std::setprecision(4) << Tc
                      << (q <= 4 ? "  (continuous)\n" : "  (first order)\n")
                      << "    T/T_c   order parameter    energy/site\n";
            for (double r : {0.9, 0.97, 1.0, 1.03, 1.1}) {
                std::mt19937 rng(seeder());
                PottsModel<int> m(std::make_unique<RegularLattice<int>>(std::vector<int>(L * L, 0), std::vector<int>{L, L}),
                                  &rng, q, r * Tc);
                std::vector<double> energy;
                const auto s = m.sample_stationary([&energy](const Model<int>& model) {
                    const auto& p = static_cast<const PottsModel<int>&>(model);
                    energy.push_back(p.energy_per_site());
                    return p.order_parameter();
                }, 500, 3000, 1);
                const double e = stats::moments(energy).m1;
                const double e_err = stats::jackknife_error(energy, 20, [](const stats::Moments& mo) { return mo.m1; });
                std::cout << "    " << std::setprecision(2) << r << "    " << std::setprecision(3) << s.mean << " +- "
                          << s.mean_err << "    " << e << " +- " << e_err << '\n';
                csv.row(q, r * Tc, s.mean, s.mean_err, e, e_err);
            }
        }
        std::cout << "  (q = 8: order and energy jump across T_c -- the latent heat of a first-order\n"
                     "   transition; near T_c a finite lattice can flip between the two phases)\n";
    }
    return 0;
}
