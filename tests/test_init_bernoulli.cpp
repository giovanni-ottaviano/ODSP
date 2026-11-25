#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "DriverTestHelpers.hpp"
#include "Utils.hpp"
#include "driver/Driver.hpp"

// --init bernoulli --up p: independent +/-1 opinions, each +1 with probability p
// (an infinite-temperature quench), next to --init random, which places exactly
// floor(p N) nodes at +1.

using namespace driver;
using namespace driver_test;

namespace {

Setup<int> setup_of(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    Options o = Options::from_args(static_cast<int>(argv.size()), argv.data());
    return read_setup<int>(o, 1);
}

long long ups(const std::vector<int>& s) {
    long long n = 0;
    for (int v : s) n += v == 1;
    return n;
}

} // namespace

TEST_CASE(init_bernoulli_spins_are_independent) {
    const Setup<int> s = setup_of({"--model", "majority", "--dims", "32x32", "--init", "bernoulli", "--up", "0.3"});
    const double N = 1024.0, p = 0.3;
    const int K = 2000;
    double sum = 0.0, sum2 = 0.0;
    for (int k = 0; k < K; ++k) {
        std::mt19937 rng(static_cast<unsigned>(1000 + k));
        const std::vector<int> st = s.factory(rng)->get_graph().states();
        bool pm = st.size() == 1024u;
        for (int v : st) pm = pm && (v == 1 || v == -1);
        CHECK(pm);
        const double n = static_cast<double>(ups(st));
        sum += n;
        sum2 += n * n;
    }
    const double mean = sum / K, var = sum2 / K - mean * mean;
    // binomial count: mean N p, variance N p (1 - p) (0 for the exact start)
    CHECK(std::abs(mean - N * p) < 4.0 * std::sqrt(N * p * (1 - p) / K));
    CHECK(std::abs(var / (N * p * (1 - p)) - 1.0) < 0.12);   // relative sd of the variance ~ 3%
}

TEST_CASE(init_bernoulli_same_seed_same_state) {
    const Setup<int> s = setup_of({"--model", "voter", "--dims", "16x16", "--init", "bernoulli", "--up", "0.5"});
    std::mt19937 a(7), b(7), c(8);
    const auto sa = s.factory(a)->get_graph().states();
    CHECK(sa == s.factory(b)->get_graph().states());
    CHECK(sa != s.factory(c)->get_graph().states());
}

TEST_CASE(init_random_still_exact_count) {
    const Setup<int> s = setup_of({"--model", "majority", "--dims", "32x32", "--init", "random", "--up", "0.3"});
    for (unsigned k = 0; k < 50; ++k) {
        std::mt19937 rng(k), ref(k);
        const std::vector<int> st = s.factory(rng)->get_graph().states();
        CHECK(ups(st) == static_cast<long long>(0.3 * 1024));
        CHECK(st == create_random_lattice(1024, 0.3, ref));   // the same draw as before the option existed
    }
}

TEST_CASE(init_bernoulli_in_the_driver) {
    TempFiles tmp;
    std::string out;
    CHECK(run({"--model", "majority", "--dims", "16x16", "--task", "ensemble", "--replicas", "20", "--init", "bernoulli",
               "--up", "0.5", "--observables", "stripes", "--seed", "3", "--csv", tmp.stem}, &out) == 0);
    CHECK(out.find("independent +/-1, P(+1) = 0.5") != std::string::npos);
    CHECK(run({"--model", "voter", "--graph", "complete", "--n", "50", "--task", "ensemble", "--replicas", "10",
               "--init", "bernoulli", "--up", "0.2", "--seed", "3"}) == 0);
    // the extremes are deterministic
    const Setup<int> all = setup_of({"--model", "voter", "--dims", "8x8", "--init", "bernoulli", "--up", "1"});
    std::mt19937 rng(1);
    CHECK(ups(all.factory(rng)->get_graph().states()) == 64);
    // errors: p outside [0, 1], a model whose opinions are not +/-1
    CHECK(run({"--model", "voter", "--dims", "8x8", "--init", "bernoulli", "--up", "1.5"}, &out) != 0);
    CHECK(out.find("--up") != std::string::npos);
    CHECK(run({"--model", "potts", "--q", "3", "--temperature", "1", "--dims", "8x8", "--init", "bernoulli"}, &out) != 0);
    CHECK(run({"--help", "init"}, &out) == 0);
    CHECK(out.find("bernoulli") != std::string::npos);
}
