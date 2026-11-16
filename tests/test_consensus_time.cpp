#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "DriverTestHelpers.hpp"
#include "driver/Driver.hpp"
#include "Ensemble.hpp"
#include "RegularLattice.hpp"
#include "VoterModel.hpp"

// consensus_time in prefix_replicas.csv: written only for models that can leave
// consensus (first passage), where stop_time is the cap. Also: runs that start
// in consensus, and the stdout note on runs that did not reach consensus.

using namespace driver;
namespace fs = std::filesystem;

using namespace driver_test;

TEST_CASE(consensus_time_written_for_first_passage_models) {
    TempFiles tmp;
    std::string out;
    CHECK(run({"--model", "noisy-voter", "--noise", "0.001", "--dims", "6x6", "--task", "ensemble",
               "--replicas", "30", "--steps", "2000", "--seed", "5", "--csv", tmp.stem}, &out) == 0);
    const std::vector<std::string> h = header_of(tmp.stem + "_replicas.csv");
    CHECK(h.size() > 4 && h[3] == "stop_time" && h[4] == "consensus_time");
    const auto rep = read_csv(tmp.stem + "_replicas.csv");
    double sum = 0.0;
    int n = 0;
    for (std::size_t i = 0; i < rep.at("replica").size(); ++i) {
        CHECK(rep.at("stop_time")[i] == 2000.0);                     // the cap, as before
        if (rep.at("reached_consensus")[i] == 1.0) {
            CHECK(rep.at("consensus_time")[i] >= 0.0 && rep.at("consensus_time")[i] <= 2000.0);
            sum += rep.at("consensus_time")[i];
            ++n;
        } else {
            CHECK(std::isnan(rep.at("consensus_time")[i]));
        }
    }
    CHECK(n > 0);
    // the driver's own first-passage mean, from its summary line
    const std::size_t at = out.find("First-passage time (sweeps): mean ");
    CHECK(at != std::string::npos);
    const double printed = std::stod(out.substr(at + std::string("First-passage time (sweeps): mean ").size()));
    CHECK_NEAR(sum / n, printed, 1e-3 * printed + 1e-9);
    // and the median (ceil(R/2)-th smallest, unreached last): order-sensitive per replica
    std::vector<double> times;
    for (std::size_t i = 0; i < rep.at("replica").size(); ++i)
        times.push_back(rep.at("reached_consensus")[i] == 1.0 ? rep.at("consensus_time")[i] : 1e300);
    std::sort(times.begin(), times.end());
    const std::size_t m = out.find(", median ", at);
    CHECK(m != std::string::npos);
    CHECK_NEAR(times[(times.size() + 1) / 2 - 1], std::stod(out.substr(m + 9)), 1e-9);
}

TEST_CASE(consensus_time_capped_runs_are_nan) {
    TempFiles tmp;
    CHECK(run({"--model", "noisy-voter", "--noise", "0.5", "--dims", "8x8", "--task", "ensemble",
               "--replicas", "6", "--steps", "3", "--seed", "2", "--csv", tmp.stem}) == 0);
    const auto rep = read_csv(tmp.stem + "_replicas.csv");
    int capped = 0;
    for (std::size_t i = 0; i < rep.at("replica").size(); ++i)
        if (rep.at("reached_consensus")[i] == 0.0) {
            CHECK(std::isnan(rep.at("consensus_time")[i]));
            ++capped;
        }
    CHECK(capped > 0);                      // the fixture really has runs that hit the cap
}

TEST_CASE(consensus_time_absent_for_absorbing_models) {
    for (std::vector<std::string> model : {std::vector<std::string>{"--model", "voter"},
                                           std::vector<std::string>{"--model", "majority"}}) {
        TempFiles tmp;
        std::vector<std::string> args = model;
        args.insert(args.end(), {"--dims", "6x6", "--task", "ensemble", "--replicas", "4", "--steps", "500",
                                 "--seed", "3", "--csv", tmp.stem});
        CHECK(run(args) == 0);
        const std::vector<std::string> h = header_of(tmp.stem + "_replicas.csv");
        for (const auto& name : h) CHECK(name != "consensus_time");
    }
}

// A replica that starts in consensus is absorbed at time 0: no step is taken.
TEST_CASE(consensus_at_start_takes_no_time) {
    for (const char* model : {"voter", "majority"}) {
        TempFiles tmp;
        std::string out;
        CHECK(run({"--model", model, "--init", "up", "--dims", "6x6", "--task", "ensemble", "--replicas", "3",
                   "--steps", "50", "--seed", "1", "--csv", tmp.stem}, &out) == 0);
        const auto rep = read_csv(tmp.stem + "_replicas.csv");
        for (std::size_t i = 0; i < rep.at("replica").size(); ++i) {
            CHECK(rep.at("reached_consensus")[i] == 1.0);
            CHECK(rep.at("stop_time")[i] == 0.0);
            if (std::string(model) == "majority") CHECK(rep.at("freeze_time")[i] == 0.0);
        }
        CHECK(out.find("Consensus time (sweeps): mean 0 ") != std::string::npos);
    }
    // the same in single updates
    TempFiles tmp;
    CHECK(run({"--model", "voter", "--init", "up", "--dims", "6x6", "--task", "ensemble", "--replicas", "2",
               "--time-unit", "updates", "--steps", "50", "--seed", "1", "--csv", tmp.stem}) == 0);
    const auto rep = read_csv(tmp.stem + "_replicas.csv");
    CHECK(rep.at("stop_time").size() == 2);
    for (double t : rep.at("stop_time")) CHECK(t == 0.0);
}

TEST_CASE(ensemble_consensus_at_start_takes_no_time) {
    auto all_up = [](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        auto g = std::make_unique<RegularLattice<int>>(std::vector<int>(16, 1), std::vector<int>{4, 4});
        return std::make_unique<VoterModel<int>>(std::move(g), &rng);
    };
    Ensemble<int> ens(all_up, 7);
    const auto r = ens.run(3, 10);
    CHECK(r.consensus_count == 3);
    for (long long t : r.consensus_times) CHECK(t == 0);
    for (long long t : r.stop_times) CHECK(t == 0);
    CHECK(r.mean_consensus_time_lower_bound == 0.0);
}

// Majority runs that freeze away from consensus never reach it: no "true mean >="
// bound (which would count them at the cap), and the note says why they are missing.
TEST_CASE(stdout_note_on_frozen_runs) {
    std::string out;
    TempFiles tmp;
    CHECK(run({"--model", "majority", "--dims", "16x16", "--task", "ensemble", "--replicas", "30",
               "--steps", "2000", "--seed", "4", "--csv", tmp.stem}, &out) == 0);
    const auto rep = read_csv(tmp.stem + "_replicas.csv");
    int frozen = 0, capped = 0, consensus = 0;
    for (std::size_t i = 0; i < rep.at("replica").size(); ++i) {
        consensus += rep.at("reached_consensus")[i] == 1.0;
        frozen += rep.at("absorbed")[i] == 1.0 && rep.at("reached_consensus")[i] == 0.0;
        capped += rep.at("absorbed")[i] == 0.0;
    }
    CHECK(frozen > 0 && consensus > 0 && capped == 0);   // the fixture: frozen runs, none capped
    CHECK(out.find("true mean >=") == std::string::npos);
    CHECK(out.find("biased low") == std::string::npos);
    CHECK(out.find(std::to_string(frozen) + " runs stopped in an absorbing state without consensus") != std::string::npos);
}

TEST_CASE(stdout_note_on_capped_runs) {
    // capped voter runs: the lower bound is valid and still printed
    std::string out;
    CHECK(run({"--model", "voter", "--dims", "8x8", "--task", "ensemble", "--replicas", "20", "--steps", "60",
               "--seed", "2"}, &out) == 0);
    CHECK(out.find("did not reach consensus") != std::string::npos);
    CHECK(out.find("hit the cap") != std::string::npos);
    CHECK(out.find("true mean >=") != std::string::npos);
    CHECK(out.find("absorbing state without consensus") == std::string::npos);
}
