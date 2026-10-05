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
#include "QVoterModel.hpp"
#include "NonlinearVoterModel.hpp"
#include "MajorityVoteModel.hpp"
#include "IsingModel.hpp"
#include "SznajdModel.hpp"
#include "Utils.hpp"
#include "Cli.hpp"
#include "CsvWriter.hpp"

// A tour of the model zoo, each compared with what theory predicts:
//   1) Ising model on the square lattice vs Onsager's exact magnetization;
//   2) majority-vote model with noise: Binder-cumulant crossing at noise_c = 0.075;
//   3) q-voter with independence: continuous (q = 3) vs discontinuous (q = 7)
//      transition, vs the mean-field solution;
//   4) voter model with zealots: stationary fluctuations vs the exact
//      Beta-Binomial law;
//   5) nonlinear voter: consensus time vs the nonlinearity alpha;
//   6) Sznajd vs voter: exit probability on a lattice.

namespace {

std::unique_ptr<BaseGraph<int>> all_up_lattice(int L) {
    return std::make_unique<RegularLattice<int>>(std::vector<int>(static_cast<std::size_t>(L * L), 1),
                                                 std::vector<int>{L, L});
}

// Largest root in [1/2, 1] of the mean-field stationarity condition of the
// q-voter with independence: the ordered branch reached from an ordered start
// (x = 1/2, the disordered solution, if there is no other root).
double qvoter_mf_ordered_x(int q, double p) {
    auto F = [q, p](double x) {
        return (1 - p) * ((1 - x) * std::pow(x, q) - x * std::pow(1 - x, q)) + p * (0.5 - x);
    };
    double hi = 1.0;                       // F(1) = -p/2 < 0
    for (double x = 1.0 - 1e-4; x > 0.5; x -= 1e-4) {
        if (F(x) > 0) {                    // sign change in (x, hi): bisect
            double lo = x;
            for (int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                (F(mid) > 0 ? lo : hi) = mid;
            }
            return 0.5 * (lo + hi);
        }
        hi = x;
    }
    return 0.5;
}

} // namespace

int main(int argc, char** argv) {
    const CliOptions opt = parse_cli(argc, argv,
        "Ising, majority-vote, q-voter, zealots, nonlinear voter and Sznajd models vs theory.");
    std::mt19937 seeder(opt.seed);
    std::cout << std::fixed;

    // ---- 1) Ising ----
    {
        const double Tc = 2.0 / std::log(1.0 + std::sqrt(2.0));
        std::cout << "=== 1) Ising model, 32x32 square lattice (Metropolis), T_c = " << std::setprecision(4) << Tc
                  << " ===\n   T      <|m|>          Onsager m(T)   Binder\n";
        CsvWriter csv(opt.csv_path("ising"), {"T", "abs_m", "abs_m_err", "onsager_m", "binder", "binder_err"});
        for (double T : {1.6, 2.0, 2.2, 2.27, 2.4, 2.8}) {
            std::mt19937 rng(seeder());
            IsingModel<int> m(all_up_lattice(32), &rng, T);
            const auto s = m.sample_stationary(500, 4000, 1);
            const double onsager = T < Tc ? std::pow(1.0 - std::pow(std::sinh(2.0 / T), -4.0), 0.125) : 0.0;
            std::cout << "  " << std::setprecision(2) << T << "   " << std::setprecision(4) << s.abs_mean << " +- "
                      << s.abs_mean_err << "   " << onsager << "         " << std::setprecision(3) << s.binder << '\n';
            csv.row(T, s.abs_mean, s.abs_mean_err, onsager, s.binder, s.binder_err);
        }
        std::cout << "  (near T_c a finite lattice keeps |m| > 0 and departs from the infinite-size curve)\n\n";
    }

    // ---- 2) Majority vote with noise ----
    {
        std::cout << "=== 2) Majority-vote model with noise: Binder cumulant U(noise) for L = 8, 16, 32 ===\n"
                  << "(curves for different L cross at the critical noise, 0.075 in the literature)\n"
                  << "  noise      L=8              L=16             L=32\n";
        CsvWriter csv(opt.csv_path("majority_vote"), {"L", "noise", "abs_m", "binder", "binder_err"});
        for (double noise : {0.065, 0.070, 0.075, 0.080, 0.085}) {
            std::cout << "  " << std::setprecision(3) << noise;
            for (int L : {8, 16, 32}) {
                std::mt19937 rng(seeder());
                MajorityVoteModel<int> m(all_up_lattice(L), &rng, noise);
                const auto s = m.sample_stationary(1000, 10000, 2);
                std::cout << "   " << std::setprecision(3) << s.binder << " +- " << s.binder_err;
                csv.row(L, noise, s.abs_mean, s.binder, s.binder_err);
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    // ---- 3) q-voter with independence ----
    {
        const std::size_t N = 5000;
        std::cout << "=== 3) q-voter with independence, complete graph N = " << N << ", ordered start ===\n"
                  << "(mean field: p_c = 1/3 for q = 3, continuous; for q = 7 the ordered branch\n"
                  << " ends abruptly at its spinodal: a discontinuous transition)\n"
                  << "  q   p       <|m|> sim   <|m|> MF\n";
        CsvWriter csv(opt.csv_path("qvoter"), {"q", "p", "abs_m", "abs_m_mf"});
        for (int q : {3, 7}) {
            const std::vector<double> ps = q == 3 ? std::vector<double>{0.1, 0.2, 0.28, 0.32, 0.36}
                                                  : std::vector<double>{0.05, 0.1, 0.12, 0.13, 0.14};
            for (double p : ps) {
                std::mt19937 rng(seeder());
                QVoterModel<int> m(std::make_unique<CompleteGraph<int>>(std::vector<int>(N, 1)), &rng, q, p);
                const auto s = m.sample_stationary(300, 500, 1);
                const double mf = 2.0 * qvoter_mf_ordered_x(q, p) - 1.0;
                std::cout << "  " << q << "   " << std::setprecision(2) << p << "    " << std::setprecision(3)
                          << s.abs_mean << "       " << mf << '\n';
                csv.row(q, p, s.abs_mean, mf);
            }
        }
        std::cout << "  (near a continuous transition finite-N fluctuations keep <|m|> ~ N^-1/4 > 0)\n\n";
    }

    // ---- 4) Zealots ----
    {
        const int N = 200;
        std::cout << "=== 4) Voter model with Z zealots of each opinion, complete graph, " << N << " free voters ===\n"
                  << "(exact: up-count of free voters ~ Beta-Binomial(N, Z, Z))\n"
                  << "  Z     <m^2> sim         <m^2> exact\n";
        CsvWriter csv(opt.csv_path("zealots"), {"Z", "m2", "m2_exact"});
        for (int Z : {1, 2, 5, 20}) {
            const int total = N + 2 * Z;
            std::mt19937 rng(seeder());
            std::vector<int> states = create_random_lattice(total, 0.5, rng);
            std::vector<std::size_t> zealots;
            for (int z = 0; z < 2 * Z; ++z) { states[z] = z < Z ? 1 : -1; zealots.push_back(z); }
            VoterModel<int> m(std::make_unique<CompleteGraph<int>>(states), &rng);
            m.set_frozen(zealots);
            const auto s = m.sample_stationary(500, 20000, 2);
            // Var n = N Z^2 (2Z + N) / ((2Z)^2 (2Z + 1)); <m> = 0, m = (2n - N) / total.
            const double var_n = static_cast<double>(N) * Z * Z * (2.0 * Z + N) / (4.0 * Z * Z * (2.0 * Z + 1.0));
            const double m2_exact = 4.0 * var_n / (static_cast<double>(total) * total);
            std::cout << "  " << std::setw(2) << Z << "    " << std::setprecision(4) << s.m2 << " +- "
                      << s.susceptibility_err / total << "   " << m2_exact << '\n';
            csv.row(Z, s.m2, m2_exact);
        }
        std::cout << "  (Z = 1: flat distribution of opinions; many zealots pin m near 0)\n\n";
    }

    // ---- 5) Nonlinear voter ----
    {
        const std::size_t N = 300;
        std::cout << "=== 5) Nonlinear voter, complete graph N = " << N << ": consensus time vs alpha ===\n"
                  << "  alpha   T_mean (sweeps)\n";
        CsvWriter csv(opt.csv_path("nonlinear"), {"alpha", "consensus_count", "replicas", "mean_sweeps"});
        for (double alpha : {0.9, 1.0, 1.5, 2.0, 3.0}) {
            Ensemble<int> ens([N, alpha](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
                return std::make_unique<NonlinearVoterModel<int>>(
                    std::make_unique<CompleteGraph<int>>(create_random_lattice(N, 0.5, rng)), &rng, alpha);
            }, seeder());
            const auto res = ens.run(200, 5000, true, opt.threads);
            std::cout << "  " << std::setprecision(1) << alpha << "     ";
            if (res.consensus_count == res.replicas) std::cout << std::setprecision(1) << res.mean_consensus_time << '\n';
            else std::cout << res.replicas - res.consensus_count << "/" << res.replicas << " not done by 5000 sweeps\n";
            csv.row(alpha, res.consensus_count, res.replicas, res.mean_consensus_time);
        }
        std::cout << "  (alpha = 1 is the voter model, T ~ N ln 2 = " << std::setprecision(0) << N * std::log(2.0)
                  << "; alpha < 1 resists consensus, alpha > 1 rushes to it)\n\n";
    }

    // ---- 6) Sznajd vs voter ----
    {
        std::cout << "=== 6) Exit probability on a 12x12 lattice: Sznajd vs voter ===\n"
                  << "  f0     voter    Sznajd\n";
        CsvWriter csv(opt.csv_path("exit"), {"f0", "voter", "sznajd"});
        for (double f0 : {0.3, 0.4, 0.5, 0.6, 0.7}) {
            double p[2];
            for (int model = 0; model < 2; ++model) {
                Ensemble<int> ens([f0, model](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
                    auto g = std::make_unique<RegularLattice<int>>(create_random_lattice(144, f0, rng),
                                                                   std::vector<int>{12, 12});
                    if (model == 0) return std::make_unique<VoterModel<int>>(std::move(g), &rng);
                    return std::make_unique<SznajdModel<int>>(std::move(g), &rng);
                }, seeder());
                p[model] = ens.run(400, 1000000, true, opt.threads).exit_prob_up;
            }
            std::cout << "  " << std::setprecision(1) << f0 << "    " << std::setprecision(3) << p[0] << "    " << p[1] << '\n';
            csv.row(f0, p[0], p[1]);
        }
        std::cout << "  (voter: P(+1) = f0; Sznajd: much steeper, a step at f0 = 1/2 for large lattices)\n";
    }
    return 0;
}
