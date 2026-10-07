#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "driver/Driver.hpp"

// The short metrics of the 2026-10-04 metrics review: persistence, activity,
// opinion entropy / effective number of opinions, segregation, opinion variance
// and bimodality, histograms, specific heat, time to a threshold, and cluster
// output for continuous models.

using namespace driver;
namespace fs = std::filesystem;

namespace {

std::unique_ptr<BaseGraph<int>> complete_of(std::vector<int> states) {
    return std::make_unique<CompleteGraph<int>>(std::move(states));
}

double row_value(const Row& row, const std::string& name) {
    for (const auto& c : row) if (c.first == name) return c.second;
    odsp_test::fail(__FILE__, __LINE__, "column '" + name + "' missing");
}

Plan<int> plan_of(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    Options o = Options::from_args(static_cast<int>(argv.size()), argv.data());
    return read_plan<int>(o);
}

int run_quiet(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* o = std::cout.rdbuf(out.rdbuf());
    std::streambuf* e = std::cerr.rdbuf(err.rdbuf());
    const int status = run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    return status;
}

std::string temp_prefix() {
    std::random_device rd;
    return (fs::temp_directory_path() / ("odsp_metrics_" + std::to_string(rd()))).string();
}

std::size_t line_count(const std::string& path) {   // data lines, without the provenance
    const std::string text = read_without_comments(path);
    return static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n'));
}

} // namespace

// ------------------------------------------------------------- persistence

TEST_CASE(metric_persistence_counts_nodes_that_never_changed) {
    std::vector<int> states(25, 1);
    states[12] = -1;   // a lone minority spin: majority rule flips exactly this node
    std::mt19937 rng(1);
    MajorityModel<int> m(std::make_unique<RegularLattice<int>>(states, std::vector<int>{5, 5}), &rng);
    CHECK(m.persistence() == 1.0);
    CHECK(m.advance(100));
    CHECK_NEAR(m.persistence(), 24.0 / 25.0, 1e-15);
    m.reset_persistence();
    CHECK(m.persistence() == 1.0);
}

TEST_CASE(metric_persistence_never_increases) {
    std::mt19937 rng(2);
    NoisyVoterModel<int> m(complete_of(create_random_lattice(100, 0.5, rng)), &rng, 0.05);
    double previous = 1.0;
    for (int s = 0; s < 50; ++s) {
        m.advance(1, true, false);
        CHECK(m.persistence() <= previous);
        previous = m.persistence();
    }
    CHECK(previous < 0.5);
}

// ---------------------------------------------------------------- activity

TEST_CASE(metric_activity_of_noisy_voter_at_full_noise_is_one_half) {
    // Every update draws a fresh +/-1 opinion, which differs from the old one
    // with probability exactly 1/2.
    std::mt19937 rng(3);
    NoisyVoterModel<int> m(complete_of(create_random_lattice(200, 0.5, rng)), &rng, 1.0);
    m.advance(200, true, false);
    const double activity = static_cast<double>(m.get_changes()) / static_cast<double>(m.get_updates());
    CHECK_NEAR(activity, 0.5, 4.0 * std::sqrt(0.25 / m.get_updates()));
    VoterModel<int> frozen(complete_of(std::vector<int>(20, 1)), &rng);
    frozen.advance(10, true, false);
    CHECK(frozen.get_changes() == 0);
}

TEST_CASE(metric_activity_observable_is_per_interval_or_cumulative) {
    std::mt19937 rng(4);
    ModelSpec spec;
    spec.name = "noisy-voter";
    NoisyVoterModel<int> m(complete_of(create_random_lattice(100, 0.5, rng)), &rng, 1.0);
    const auto per_interval = make_observable<int>("activity", spec, /*per_interval=*/true);
    const auto cumulative = make_observable<int>("activity", spec);
    m.advance(50, true, false);
    CHECK_NEAR(per_interval(m), 0.5, 0.03);   // since start
    CHECK_NEAR(cumulative(m), 0.5, 0.03);
    m.advance(50, true, false);
    CHECK_NEAR(per_interval(m), 0.5, 0.03);   // since the previous reading
}

// ---------------------------------------------- entropy, effective opinions

TEST_CASE(metric_opinion_entropy_and_effective_number) {
    std::mt19937 rng(5);
    VoterModel<int> half(complete_of({0, 0, 1, 1}), &rng);
    CHECK_NEAR(half.opinion_entropy(), std::log(2.0), 1e-15);
    CHECK_NEAR(half.effective_number_of_opinions(), 2.0, 1e-12);
    VoterModel<int> four(complete_of({0, 1, 2, 3}), &rng);
    CHECK_NEAR(four.effective_number_of_opinions(), 4.0, 1e-12);
    VoterModel<int> skewed(complete_of({0, 0, 0, 0, 0, 0, 0, 0, 0, 1}), &rng);
    CHECK(skewed.effective_number_of_opinions() > 1.0 && skewed.effective_number_of_opinions() < 2.0);
    VoterModel<int> consensus(complete_of({3, 3, 3}), &rng);
    CHECK(consensus.opinion_entropy() == 0.0);
    CHECK(consensus.effective_number_of_opinions() == 1.0);
}

// ------------------------------------------------------------- segregation

TEST_CASE(metric_segregation_index) {
    std::mt19937 rng(6);
    // Complete graph: rho equals its random-placement value exactly, S = 0.
    VoterModel<int> mixed(complete_of(create_balanced_opinions(30, 3, rng)), &rng);
    CHECK_NEAR(mixed.segregation(), 0.0, 1e-12);
    // Ring of 10, two domains of 5: rho = 2/10; random placement gives
    // 1 - (5*4 + 5*4)/(10*9) = 5/9, so S = 1 - 0.2/(5/9) = 0.64.
    VoterModel<int> ring(std::make_unique<RegularLattice<int>>(std::vector<int>{1, 1, 1, 1, 1, -1, -1, -1, -1, -1},
                                                               std::vector<int>{10}), &rng);
    CHECK_NEAR(ring.segregation(), 0.64, 1e-12);
    // Consensus: no mixing at all, reported as 1.
    VoterModel<int> consensus(complete_of(std::vector<int>(10, 1)), &rng);
    CHECK(consensus.segregation() == 1.0);
}

// ------------------------------------------------ variance and bimodality

TEST_CASE(metric_bimodality_coefficient_on_known_shapes) {
    std::mt19937 rng(7);
    std::vector<double> two_point(1000), uniform(200000), normal(200000);
    for (std::size_t i = 0; i < two_point.size(); ++i) two_point[i] = i % 2;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> g(0.0, 1.0);
    for (double& x : uniform) x = u(rng);
    for (double& x : normal) x = g(rng);
    CHECK(stats::bimodality_coefficient(two_point) > 0.95);                 // clearly bimodal
    CHECK_NEAR(stats::bimodality_coefficient(uniform), 5.0 / 9.0, 0.01);    // the 5/9 reference
    CHECK_NEAR(stats::bimodality_coefficient(normal), 1.0 / 3.0, 0.01);     // unimodal
}

TEST_CASE(metric_opinion_variance_and_bimodality_of_a_model) {
    std::mt19937 rng(8);
    VoterModel<double> m(std::make_unique<CompleteGraph<double>>(std::vector<double>{0.0, 0.0, 1.0, 1.0}), &rng);
    CHECK_NEAR(m.opinion_variance(), 0.25, 1e-15);
    std::vector<double> x(1000);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = i % 2;
    VoterModel<double> big(std::make_unique<CompleteGraph<double>>(x), &rng);
    CHECK_NEAR(big.bimodality_coefficient(), stats::bimodality_coefficient(x), 1e-12);
}

// ---------------------------------------------------------------- histogram

TEST_CASE(metric_histogram_counts) {
    const std::vector<std::size_t> h = stats::histogram({0.0, 0.1, 0.5, 0.9, 1.0}, 2, 0.0, 1.0);
    CHECK(h == (std::vector<std::size_t>{2, 3}));   // the upper edge belongs to the last bin
    CHECK(stats::histogram({2.0, 2.0}, 4, 2.0, 2.0) == (std::vector<std::size_t>{2, 0, 0, 0}));   // zero width
}

TEST_CASE(metric_stationary_writes_histogram) {
    const std::string prefix = temp_prefix();
    CHECK(run_quiet({"--model", "noisy-voter", "--noise", "0.5", "--graph", "complete", "--n", "50", "--task",
                     "stationary", "--samples", "500", "--bins", "10", "--seed", "1", "--csv", prefix}) == 0);
    CHECK(line_count(prefix + "_histogram.csv") == 11);   // header + 10 bins
    std::error_code ec;
    for (const char* s : {"_histogram.csv", "_samples.csv", "_config.txt"}) fs::remove(prefix + s, ec);
}

// ------------------------------------------------------------ specific heat

TEST_CASE(metric_specific_heat_matches_exact_enumeration) {
    // C = N (<e^2> - <e>^2) / T^2 per site, for the periodic 3x3 Ising lattice.
    const double T = 2.5;
    RegularLattice<int> geometry(std::vector<int>(9, 1), {3, 3});
    double Z = 0.0, s1 = 0.0, s2 = 0.0;
    for (int config = 0; config < 512; ++config) {
        auto spin = [config](std::size_t i) { return (config >> i) & 1 ? 1 : -1; };
        double E = 0.0;
        for (std::size_t i = 0; i < 9; ++i)
            for (std::size_t j : geometry.neighbours(i))
                if (j > i) E -= spin(i) * spin(j);
        const double w = std::exp(-E / T), e = E / 9.0;
        Z += w; s1 += w * e; s2 += w * e * e;
    }
    const double exact = 9.0 * (s2 / Z - (s1 / Z) * (s1 / Z)) / (T * T);

    Plan<int> p = plan_of({"--model", "ising", "--temperature", "2.5", "--dims", "3x3", "--task", "stationary",
                           "--burn-in", "1000", "--samples", "100000", "--seed", "9"});
    const Row row = task_stationary<int>(p.params, p.setup, 10, "", false);
    CHECK_NEAR(row_value(row, "specific_heat"), exact, 4.0 * row_value(row, "specific_heat_err"));
}

// ------------------------------------------------------ time to a threshold

TEST_CASE(metric_threshold_time_at_one_is_the_consensus_time) {
    Plan<int> p = plan_of({"--model", "voter", "--graph", "complete", "--n", "30", "--task", "ensemble",
                           "--replicas", "40", "--threshold", "1", "--seed", "11"});
    const Row row = task_ensemble<int>(p.params, p.setup, 12, 1, "", false);
    CHECK(row_value(row, "threshold_fraction") == 1.0);
    CHECK_NEAR(row_value(row, "mean_threshold_time"), row_value(row, "mean_consensus_time"), 1e-12);
    // A threshold already met at the start is reached at time 0.
    Plan<int> q = plan_of({"--model", "voter", "--graph", "complete", "--n", "30", "--task", "ensemble",
                           "--replicas", "10", "--threshold", "0.4", "--seed", "11"});
    CHECK(row_value(task_ensemble<int>(q.params, q.setup, 12, 1, "", false), "mean_threshold_time") == 0.0);
}

TEST_CASE(metric_threshold_needs_discrete_opinions) {
    CHECK(run_quiet({"--model", "deffuant", "--epsilon", "0.2", "--graph", "complete", "--n", "20", "--task",
                     "ensemble", "--threshold", "0.9"}) == 2);
    CHECK(run_quiet({"--model", "voter", "--task", "ensemble", "--threshold", "1.5"}) == 2);
}

// ------------------------------------------- clusters of continuous models

TEST_CASE(metric_continuous_ensemble_writes_clusters) {
    const std::string prefix = temp_prefix();
    CHECK(run_quiet({"--model", "deffuant", "--epsilon", "0.3", "--graph", "complete", "--n", "200", "--task",
                     "ensemble", "--replicas", "5", "--seed", "13", "--csv", prefix}) == 0);
    std::istringstream in(read_without_comments(prefix + "_clusters.csv"));
    std::string header;
    std::getline(in, header);
    CHECK(header == "replica,position,size,min,max");
    CHECK(line_count(prefix + "_clusters.csv") >= 6);   // at least one cluster per replica
    std::error_code ec;
    for (const char* s : {"_clusters.csv", "_replicas.csv", "_config.txt"}) fs::remove(prefix + s, ec);
}

// ------------------------------------------------- observable availability

TEST_CASE(metric_observables_are_available_where_meaningful) {
    ModelSpec voter;  voter.name = "voter";
    ModelSpec hk;     hk.name = "hk";
    for (const char* name : {"persistence", "activity", "entropy", "effective_opinions", "segregation", "variance",
                             "bimodality"})
        make_observable<int>(name, voter);
    for (const char* name : {"persistence", "activity", "variance", "bimodality"}) make_observable<double>(name, hk);
    CHECK_THROWS_AS(make_observable<double>("entropy", hk), OptionError);
    CHECK_THROWS_AS(make_observable<double>("segregation", hk), OptionError);
}
