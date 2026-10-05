#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>

#include "BaseGraph.hpp"
#include "RegularLattice.hpp"
#include "CompleteGraph.hpp"
#include "VoterModel.hpp"
#include "MajorityModel.hpp"
#include "RejectionFreeVoterModel.hpp"
#include "Utils.hpp"
#include "Cli.hpp"
#include "CsvWriter.hpp"

// Any Model runs on any BaseGraph. Templated on the model type so we can drive
// the full model x topology matrix with one function.
template <typename ModelT>
static void run(std::unique_ptr<BaseGraph<int>> graph, const char* model,
                const char* topo, std::mt19937& rng, long long steps, CsvWriter& csv) {
    std::cout << "=== " << model << " on " << topo
              << " (" << graph->size() << " nodes) ===\n";
    const std::size_t nodes = graph->size();
    ModelT m(std::move(graph), &rng);
    std::cout << "Initial m/site = " << m.get_magnetization_per_site()
              << ", rho = " << m.get_active_link_density() << '\n';
    m.run_montecarlo(steps, /*log%=*/50, /*use_mc_steps=*/true);
    std::cout << "Totals: " << m.get_mc_steps() << " sweeps, "
              << m.get_updates() << " updates.\n\n";
    csv.row(model, topo, nodes, m.get_mc_steps(), m.get_updates(),
            m.is_consensus_reached() ? 1 : 0,
            m.get_magnetization_per_site(), m.get_active_link_density());
}

// Sample the active-link density sweep-by-sweep to expose the coarsening curve.
template <typename ModelT>
static void trace_rho(std::unique_ptr<BaseGraph<int>> graph, const char* label,
                      std::mt19937& rng, long long max_sweeps, CsvWriter& csv) {
    ModelT m(std::move(graph), &rng);
    std::cout << label << "\n  sweep :  rho\n";
    long long next = 1;   // sample at 0,1,2,4,8,... (log-spaced)
    for (long long s = 0; s <= max_sweeps; ++s) {
        if (s == 0 || s == next) {
            std::cout << "  " << std::setw(5) << s << " :  "
                      << std::fixed << std::setprecision(4)
                      << m.get_active_link_density() << '\n';
            csv.row(label, s, m.get_active_link_density());
            if (s == next) next *= 2;
        }
        if (m.is_consensus_reached()) { std::cout << "  (consensus at sweep " << s << ")\n"; break; }
        m.advance(1, /*use_mc_steps=*/true);   // one silent sweep
    }
    std::cout << '\n';
}

// Same voter dynamics, two algorithms: time to consensus on a 1D ring, where
// almost every plain update is wasted late in the run.
template <typename ModelT>
static void time_to_consensus(const char* label, std::size_t n, std::mt19937& rng, CsvWriter& csv) {
    ModelT m(std::make_unique<RegularLattice<int>>(create_random_lattice(n, 0.5, rng), std::vector<int>{static_cast<int>(n)}),
             &rng);
    const auto start = std::chrono::steady_clock::now();
    m.advance(100000000LL);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "  " << std::setw(15) << std::left << label << std::right
              << std::setw(9) << m.get_mc_steps() << " sweeps  " << std::setprecision(1) << std::setw(8) << ms << " ms\n";
    csv.row(label, n, m.get_mc_steps(), ms);
}

int main(int argc, char** argv) {
    const CliOptions opt = parse_cli(argc, argv,
        "Voter and majority models on a 2D lattice and a complete graph.");
    std::mt19937 rng(opt.seed);

    CsvWriter runs_csv(opt.csv_path("runs"), {"model", "topology", "nodes", "sweeps", "updates",
                                              "consensus", "final_m_per_site", "final_rho"});
    CsvWriter rho_csv(opt.csv_path("rho"), {"model", "sweep", "rho"});
    CsvWriter timing_csv(opt.csv_path("timing"), {"algorithm", "nodes", "sweeps", "ms"});

    const std::vector<int> dims = {24, 24};
    const std::size_t N = 24 * 24;
    const long long cap = 20000;

    auto lattice  = [&]{ return std::make_unique<RegularLattice<int>>(create_random_lattice(N, 0.5, rng), dims); };
    auto complete = [&]{ return std::make_unique<CompleteGraph<int>>(create_random_lattice(N, 0.5, rng)); };

    std::cout << "--- model x topology matrix ---\n\n";
    run<VoterModel<int>>   (lattice(),  "Voter",    "lattice 24x24", rng, cap, runs_csv);
    run<VoterModel<int>>   (complete(), "Voter",    "complete graph", rng, cap, runs_csv);
    run<MajorityModel<int>>(lattice(),  "Majority", "lattice 24x24", rng, cap, runs_csv);
    run<MajorityModel<int>>(complete(), "Majority", "complete graph", rng, cap, runs_csv);

    std::cout << "--- active-link density rho(t) on a 24x24 lattice ---\n";
    std::cout << "(voter coarsens slowly toward rho=0; majority drops fast, then either\n"
                 " orders (rho->0) or freezes at rho>0, depending on the initial condition)\n\n";
    trace_rho<VoterModel<int>>   (lattice(), "Voter",    rng, 4096, rho_csv);
    trace_rho<MajorityModel<int>>(lattice(), "Majority", rng, 4096, rho_csv);

    std::cout << "--- voter on a 1D ring of 512 nodes: plain vs rejection-free ---\n";
    std::cout << "(identical dynamics; consensus times differ only by randomness)\n";
    for (int rep = 0; rep < 3; ++rep) {
        time_to_consensus<VoterModel<int>>             ("plain",          512, rng, timing_csv);
        time_to_consensus<RejectionFreeVoterModel<int>>("rejection-free", 512, rng, timing_csv);
    }
    std::cout << '\n';

    if (opt.csv_enabled())
        std::cout << "CSV written: " << runs_csv.path() << ", " << rho_csv.path() << ", "
                  << timing_csv.path() << '\n';
    return 0;
}
