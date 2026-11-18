#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "driver/Driver.hpp"

// Stationary runs state how uncertain their error bars are: an error estimated
// from k independent groups (chains, or jackknife blocks of one chain) is
// itself uncertain by about 1/sqrt(2 (k - 1)).

using namespace driver;

namespace {

std::string output_of(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* o = std::cout.rdbuf(out.rdbuf());
    std::streambuf* e = std::cerr.rdbuf(err.rdbuf());
    const int status = run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    CHECK(status == 0);
    return out.str();
}

std::size_t count(const std::string& text, const std::string& part) {
    std::size_t n = 0;
    for (std::size_t p = text.find(part); p != std::string::npos; p = text.find(part, p + 1)) ++n;
    return n;
}

} // namespace

TEST_CASE(error_bar_uncertainty_formula) {
    CHECK(std::abs(error_bar_uncertainty(10) - 1.0 / std::sqrt(18.0)) < 1e-15);   // 24%
    CHECK(std::abs(error_bar_uncertainty(20) - 1.0 / std::sqrt(38.0)) < 1e-15);   // 16%
    CHECK(std::isnan(error_bar_uncertainty(1)));
}

TEST_CASE(error_bar_uncertainty_is_printed) {
    const std::vector<std::string> base = {"--model", "ising", "--temperature", "3", "--dims", "8x8", "--task", "stationary",
                                           "--burn-in", "10", "--samples", "200", "--seed", "1"};
    std::vector<std::string> chains = base;
    chains.insert(chains.end(), {"--replicas", "10"});
    const std::string c = output_of(chains);
    CHECK(count(c, "error bars from 10 chains are themselves uncertain by about +-24%") == 1);
    CHECK(count(c, "more --replicas") == 1);

    const std::string one = output_of(base);   // one chain, 20 jackknife blocks
    CHECK(count(one, "error bars from 20 jackknife blocks are themselves uncertain by about +-16%") == 1);

    std::vector<std::string> sweep = chains;
    sweep.insert(sweep.end(), {"--sweep", "temperature=2.5,3,3.5", "--threads", "3"});
    CHECK(count(output_of(sweep), "error bars from 10 chains are themselves uncertain by about +-24%") == 1);   // once
}

// The short-block WARNING concerns the jackknife within one chain. With several
// chains the errors are leave-one-chain-out and do not depend on the block
// length, so it is not printed then; with one chain both runs and sweeps still
// warn. Ising at T_c with interval 1 has tau_abs of several samples, longer than
// a tenth of the 10-sample blocks used here.
TEST_CASE(short_block_warning_only_for_one_chain) {
    const std::vector<std::string> base = {"--model", "ising", "--temperature", "2.27", "--dims", "16x16", "--task", "stationary",
                                           "--burn-in", "200", "--samples", "200", "--interval", "1", "--seed", "3"};
    const std::string block_warning = "not much longer than the autocorrelation time";
    CHECK(count(output_of(base), block_warning) == 1);

    std::vector<std::string> chains = base;
    chains.insert(chains.end(), {"--replicas", "4"});
    const std::string c = output_of(chains);
    CHECK(count(c, block_warning) == 0);

    std::vector<std::string> sweep = base;
    sweep.insert(sweep.end(), {"--sweep", "temperature=2.27,2.3", "--threads", "2"});
    CHECK(count(output_of(sweep), "the jackknife blocks are not much longer") == 1);
    std::vector<std::string> sweep_chains = chains;
    sweep_chains.insert(sweep_chains.end(), {"--sweep", "temperature=2.27,2.3", "--threads", "2"});
    CHECK(count(output_of(sweep_chains), "the jackknife blocks are not much longer") == 0);
}

// The rare-sign-change NOTE: with one chain, when the jackknife blocks are long
// against tau(|x|) but short against tau(x) (block < 10 tau(x)); with several
// chains, when each chain is shorter than 50 tau(x) (estimated per chain, it
// saturates near chain length / 15 when too short: see task_stationary): chains
// started alike (ordered starts) are then biased the same way and
// leave-one-chain-out errors do not see it. Ising 8x8 with interval 1 from
// all-up starts: at T = 2.0 x flips sign every ~1e3 samples (tau(|x|) ~ 6):
// one chain of 20000 samples (blocks of 1000) gets tau(x) ~ 950 and 4 chains of
// 5000 get ~140 (both flagged); at T = 3 tau(x) ~ 12 (neither flagged).
TEST_CASE(sign_note_one_and_several_chains) {
    const std::vector<std::string> base = {"--model", "ising", "--dims", "8x8", "--init", "up", "--task", "stationary",
                                           "--burn-in", "200", "--interval", "1", "--seed", "3"};
    const std::string note = "changes sign only rarely";
    auto with = [&](std::vector<std::string> extra) {
        std::vector<std::string> args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        return output_of(args);
    };

    const std::string one_rare = with({"--temperature", "2.0", "--samples", "20000"});
    CHECK(count(one_rare, note) == 1);
    CHECK(count(one_rare, "|mean|, abs_susceptibility and Binder are fine") == 1);
    CHECK(count(with({"--temperature", "3", "--samples", "20000"}), note) == 0);

    const std::string chains_rare = with({"--temperature", "2.0", "--samples", "5000", "--replicas", "4"});
    CHECK(count(chains_rare, note) == 1);
    CHECK(count(chains_rare, "within a chain") == 1);
    CHECK(count(chains_rare, "the plain mean and N var(x), and their errors, may be biased") == 1);
    CHECK(count(chains_rare, "|mean|, abs_susceptibility and Binder are fine") == 1);
    CHECK(count(with({"--temperature", "3", "--samples", "5000", "--replicas", "4"}), note) == 0);
}

// Short chains alone do not make the several-chain sign NOTE: it also needs
// tau(x) > 2 tau(|x|). An observable that never changes sign (the Potts order
// parameter, >= 0; the Ising energy, < 0) has tau(x) = tau(|x|), here ~10 samples
// against chains of 200 (< 50 tau), and must not get it.
TEST_CASE(sign_note_absent_when_x_keeps_its_sign) {
    const std::string note = "changes sign only rarely";
    const std::string potts = output_of({"--model", "potts", "--q", "3", "--dims", "16x16", "--temperature", "0.995",
                                         "--task", "stationary", "--observable", "order", "--samples", "200",
                                         "--replicas", "4", "--seed", "3"});
    CHECK(count(potts, note) == 0);
    const std::string energy = output_of({"--model", "ising", "--dims", "16x16", "--temperature", "2.27",
                                          "--task", "stationary", "--observable", "energy", "--samples", "200",
                                          "--replicas", "4", "--seed", "3"});
    CHECK(count(energy, note) == 0);
}
