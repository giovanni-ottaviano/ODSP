#include <chrono>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "driver/Driver.hpp"

// Regression tests for the bugs found by the 2026-10-04 code review.

using namespace driver;

namespace {

int run_status(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* old_out = std::cout.rdbuf(out.rdbuf());
    std::streambuf* old_err = std::cerr.rdbuf(err.rdbuf());
    const int status = run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    return status;
}

std::unique_ptr<BaseGraph<int>> complete_of(std::vector<int> states) {
    return std::make_unique<CompleteGraph<int>>(std::move(states));
}

} // namespace

// ---- bug 1: consensus statistics for models that can leave consensus ----

TEST_CASE(regression_ensemble_first_passage_for_non_absorbing_models) {
    // Noisy voter: consensus is visited and left again. The ensemble must
    // record the FIRST time each replica reaches it, not "in consensus at the
    // step cap", and must not report fixation (nothing is fixed).
    Ensemble<int> ens([](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        return std::make_unique<NoisyVoterModel<int>>(complete_of(create_random_lattice(10, 0.5, rng)), &rng, 0.3);
    }, /*seed=*/1);
    const auto res = ens.run(50, /*max_steps=*/200);
    CHECK(res.consensus_count > 25);                  // most replicas hit consensus at some point
    CHECK(res.mean_consensus_time < 100.0);           // first-passage times, not the cap (200)
    for (long long t : res.consensus_times) CHECK(t <= 200);
    CHECK(res.fixation_counts.empty());
    CHECK(std::isnan(res.exit_prob_up));
    CHECK(res.first_passage);
}

TEST_CASE(regression_first_passage_at_time_zero) {
    // Starting in consensus: the first passage is at time 0, and the replica
    // still runs to the cap (its final state is measured there).
    Ensemble<int> ens([](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        return std::make_unique<NoisyVoterModel<int>>(complete_of(std::vector<int>(10, 1)), &rng, 0.2);
    }, /*seed=*/2);
    const auto res = ens.run(10, /*max_steps=*/30);
    CHECK(res.consensus_count == 10);
    for (long long t : res.consensus_times) CHECK(t == 0);
    for (long long t : res.stop_times) CHECK(t == 30);
}

TEST_CASE(regression_absorbing_models_still_report_fixation) {
    Ensemble<int> ens([](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        return std::make_unique<VoterModel<int>>(complete_of(create_random_lattice(10, 0.5, rng)), &rng);
    }, /*seed=*/3);
    const auto res = ens.run(50, 100000);
    CHECK(!res.first_passage);
    CHECK(res.consensus_count == 50);
    CHECK(!res.fixation_counts.empty());
    CHECK(!std::isnan(res.exit_prob_up));
}

// ---- bug 2: opinions outside a model's opinion set ----

TEST_CASE(regression_binary_models_reject_non_binary_states) {
    std::mt19937 rng(4);
    const std::vector<int> bad = {1, -1, 0, 1};
    CHECK_THROWS_AS(IsingModel<int>(complete_of(bad), &rng, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(QVoterModel<int>(complete_of(bad), &rng, 2), std::invalid_argument);
    CHECK_THROWS_AS(NonlinearVoterModel<int>(complete_of(bad), &rng, 2.0), std::invalid_argument);
    CHECK_THROWS_AS(MajorityVoteModel<int>(complete_of(bad), &rng, 0.1), std::invalid_argument);
    IsingModel<int> ok(complete_of({1, -1, -1, 1}), &rng, 1.0);   // +/-1 is fine
    CHECK(ok.get_graph().size() == 4);
}

TEST_CASE(regression_driver_checks_initial_and_zealot_opinions) {
    // A 3-opinion noisy voter cannot start from +/-1 states.
    CHECK(run_status({"--model", "noisy-voter", "--noise", "0.1", "--opinions", "3", "--init", "random",
                      "--graph", "complete", "--n", "20", "--steps", "1"}) == 2);
    CHECK(run_status({"--model", "noisy-voter", "--noise", "0.1", "--opinions", "3", "--init", "all", "--value", "2",
                      "--graph", "complete", "--n", "20", "--steps", "1"}) == 0);
    // Zealots must hold an opinion the model knows.
    CHECK(run_status({"--model", "ising", "--temperature", "1", "--zealots", "5:0", "--dims", "8x8", "--steps", "1"}) == 2);
    CHECK(run_status({"--model", "qvoter", "--q", "2", "--zealots", "5:7", "--dims", "8x8", "--steps", "1"}) == 2);
    CHECK(run_status({"--model", "potts", "--q", "3", "--temperature", "1", "--zealots", "5:3", "--dims", "8x8",
                      "--steps", "1"}) == 2);
    CHECK(run_status({"--model", "voter", "--zealots", "5:7", "--dims", "8x8", "--steps", "1"}) == 0);   // any integer
    CHECK(run_status({"--model", "ising", "--temperature", "1", "--zealots", "5:-1", "--dims", "8x8", "--steps", "1"}) == 0);
}

// ---- bug 3: Ising consensus at T = 0 with isolated nodes ----

TEST_CASE(regression_ising_zero_temperature_isolated_nodes) {
    std::mt19937 rng(5);
    IsingModel<int> isolated(std::make_unique<AdjacencyGraph<int>>(std::vector<int>{1, 1, 1}, EdgeList{{0, 1}}),
                             &rng, 0.0);
    CHECK(!isolated.consensus_is_absorbing());   // node 2 has dE = 0 and flips freely
    IsingModel<int> connected(std::make_unique<AdjacencyGraph<int>>(std::vector<int>{1, 1, 1}, EdgeList{{0, 1}, {1, 2}}),
                              &rng, 0.0);
    CHECK(connected.consensus_is_absorbing());
}

// ---- bug 4: seeds outside 32 bits ----

TEST_CASE(regression_driver_rejects_out_of_range_seeds) {
    CHECK(run_status({"--model", "voter", "--dims", "4x4", "--steps", "1", "--seed", "4294967296"}) == 2);
    CHECK(run_status({"--model", "voter", "--dims", "4x4", "--steps", "1", "--seed", "-1"}) == 2);
    CHECK(run_status({"--model", "voter", "--dims", "4x4", "--steps", "1", "--seed", "4294967295"}) == 0);
}

// ---- bug 5: the "sweeps" observable with single updates ----

TEST_CASE(regression_sweeps_observable_counts_single_updates) {
    std::mt19937 rng(6);
    VoterModel<int> m(complete_of(std::vector<int>(10, 1)), &rng);
    m.advance(25, /*use_mc_steps=*/false, false);
    ModelSpec spec;
    spec.name = "voter";
    CHECK_NEAR(make_observable<int>("sweeps", spec)(m), 2.5, 1e-12);
}

// ---- bug 7: complete graph with many opinions ----

TEST_CASE(regression_complete_graph_many_opinions_is_fast) {
    // Count queries and updates must not scan all opinions. With 50000
    // distinct opinions the linear-scan version needs ~2 s for one sweep
    // (cost ~ N^2), the O(1) version milliseconds; the bound separates them
    // with a wide margin, also in the sanitizer build.
    std::mt19937 rng(7);
    VoterModel<int> m(std::make_unique<CompleteGraph<int>>(create_distinct_opinions(50000)), &rng);
    const auto start = std::chrono::steady_clock::now();
    m.advance(1, true, false);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    CHECK(seconds < 0.3);
    CHECK(m.number_of_opinions() < 50000);
}

// ---- bug 8: sweeping a topology option rebuilds the network ----

TEST_CASE(regression_sweep_over_graph_size_rebuilds_network) {
    std::random_device rd;
    const std::string prefix = (std::filesystem::temp_directory_path() / ("odsp_reg_" + std::to_string(rd()))).string();
    CHECK(run_status({"--model", "voter", "--graph", "complete", "--task", "ensemble", "--replicas", "5",
                      "--sweep", "n=10,40", "--observables", "m", "--seed", "8", "--csv", prefix}) == 0);
    std::ifstream in(prefix + "_point1_replicas.csv");
    std::string header, row;
    std::getline(in, header);
    CHECK(static_cast<bool>(std::getline(in, row)));
    std::error_code ec;
    for (const char* suffix : {"_config.txt", "_sweep.csv", "_point0_replicas.csv", "_point0_fixation.csv",
                               "_point1_replicas.csv", "_point1_fixation.csv"})
        std::filesystem::remove(prefix + suffix, ec);
}
