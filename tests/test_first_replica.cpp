#include <fstream>
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
#include "Utils.hpp"
#include "VoterModel.hpp"

// --first-replica k: replicas k..k+R-1 of the ensemble, so one ensemble split
// across several jobs gives, merged, exactly the replicas of a single run.

using namespace driver_test;

namespace {

const std::vector<std::string> kBase = {"--model", "voter", "--dims", "6x6", "--task", "ensemble",
                                        "--steps", "300", "--observables", "abs_m,rho", "--seed", "11"};

std::vector<std::string> with(std::vector<std::string> a, const std::vector<std::string>& extra) {
    a.insert(a.end(), extra.begin(), extra.end());
    return a;
}

// The data lines of a CSV (no preamble, no header).
std::vector<std::string> data_lines(const std::string& path) {
    std::istringstream in(read_without_comments(path));
    std::string line;
    std::getline(in, line);
    std::vector<std::string> lines;
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}

// Column names of a states file, from its commented "# node,..." header.
std::vector<std::string> states_header(const std::string& path) {
    std::ifstream in(path);
    std::string line, cell;
    while (std::getline(in, line) && line.rfind("# node,", 0) != 0) {}
    std::vector<std::string> names;
    std::stringstream hs(line);
    while (std::getline(hs, cell, ',')) names.push_back(cell);
    return names;
}

std::vector<std::string> concat(std::vector<std::string> a, const std::vector<std::string>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

}  // namespace

TEST_CASE(ensemble_first_replica_continues_the_seed_sequence) {
    auto factory = [](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        auto g = std::make_unique<RegularLattice<int>>(create_random_lattice(36, 0.5, rng), std::vector<int>{6, 6});
        return std::make_unique<VoterModel<int>>(std::move(g), &rng);
    };
    const auto all = Ensemble<int>(factory, 5).run(9, 300);
    const auto head = Ensemble<int>(factory, 5, 0).run(4, 300);
    const auto tail = Ensemble<int>(factory, 5, 4).run(5, 300);
    std::vector<long long> stops = head.stop_times;
    stops.insert(stops.end(), tail.stop_times.begin(), tail.stop_times.end());
    CHECK(stops == all.stop_times);
    CHECK(Ensemble<int>(factory, 5, 4).run(5, 300).stop_times == tail.stop_times);   // reproducible
}

TEST_CASE(first_replica_split_runs_merge_into_the_full_run) {
    TempFiles full, a, b;
    CHECK(run(with(kBase, {"--replicas", "10", "--log-times", "4", "--save-trajectories", "true",
                           "--save-states", "true", "--csv", full.stem})) == 0);
    CHECK(run(with(kBase, {"--replicas", "4", "--log-times", "4", "--save-trajectories", "true",
                           "--save-states", "true", "--csv", a.stem})) == 0);
    CHECK(run(with(kBase, {"--replicas", "6", "--first-replica", "4", "--log-times", "4", "--save-trajectories", "true",
                           "--save-states", "true", "--threads", "3", "--csv", b.stem})) == 0);
    for (const char* file : {"_replicas.csv", "_trajectories.csv"})
        CHECK(concat(data_lines(a.stem + file), data_lines(b.stem + file)) == data_lines(full.stem + file));
    // states: one column per replica, named by its index in the whole ensemble
    const auto h = states_header(b.stem + "_states.csv");
    CHECK(h.size() == 7 && h[1] == "replica4" && h[6] == "replica9");
    // replica 7's final state: the same column in both files
    auto column = [](const std::string& path, std::size_t k) {
        std::istringstream in(read_without_comments(path));
        std::string line, cell;
        std::vector<std::string> col;
        while (std::getline(in, line)) {
            std::stringstream ls(line);
            for (std::size_t j = 0; std::getline(ls, cell, ','); ++j)
                if (j == k) col.push_back(cell);
        }
        return col;
    };
    CHECK(!column(full.stem + "_states.csv", 8).empty());
    CHECK(column(full.stem + "_states.csv", 8) == column(b.stem + "_states.csv", 4));
}

TEST_CASE(first_replica_labels_domains_and_clusters) {
    TempFiles full, b;
    const std::vector<std::string> dom = {"--model", "majority", "--dims", "8x8", "--task", "ensemble", "--steps", "50",
                                          "--observables", "domains", "--seed", "3"};
    CHECK(run(with(dom, {"--replicas", "5", "--csv", full.stem})) == 0);
    CHECK(run(with(dom, {"--replicas", "2", "--first-replica", "3", "--csv", b.stem})) == 0);
    const auto f = data_lines(full.stem + "_domains.csv"), p = data_lines(b.stem + "_domains.csv");
    std::vector<std::string> tail;
    for (const auto& line : f)
        if (line.rfind("3,", 0) == 0 || line.rfind("4,", 0) == 0) tail.push_back(line);
    CHECK(!p.empty() && p == tail);

    TempFiles cf, cb;
    const std::vector<std::string> cont = {"--model", "deffuant", "--epsilon", "0.3", "--n", "40", "--graph", "complete", "--task", "ensemble",
                                           "--steps", "100", "--seed", "8"};
    CHECK(run(with(cont, {"--replicas", "3", "--csv", cf.stem})) == 0);
    CHECK(run(with(cont, {"--replicas", "1", "--first-replica", "2", "--csv", cb.stem})) == 0);
    std::vector<std::string> last;
    for (const auto& line : data_lines(cf.stem + "_clusters.csv"))
        if (line.rfind("2,", 0) == 0) last.push_back(line);
    CHECK(!last.empty() && data_lines(cb.stem + "_clusters.csv") == last);
}

TEST_CASE(first_replica_in_a_sweep_offsets_every_point) {
    TempFiles full, b;
    const std::vector<std::string> sw = with(kBase, {"--sweep", "dims=4x4,6x6"});
    CHECK(run(with(sw, {"--replicas", "5", "--csv", full.stem})) == 0);
    CHECK(run(with(sw, {"--replicas", "2", "--first-replica", "3", "--csv", b.stem})) == 0);
    for (const char* point : {"_point0_replicas.csv", "_point1_replicas.csv"}) {
        const auto f = data_lines(full.stem + point);
        CHECK(f.size() == 5);
        CHECK(data_lines(b.stem + point) == std::vector<std::string>(f.begin() + 3, f.end()));
    }
}

TEST_CASE(first_replica_validation) {
    std::string out;
    CHECK(run(with(kBase, {"--replicas", "3", "--first-replica", "-1"}), &out) == 2);
    CHECK(out.find("--first-replica") != std::string::npos);
    CHECK(run({"--model", "ising", "--temperature", "2.5", "--dims", "6x6", "--task", "stationary", "--samples", "10",
               "--first-replica", "2"},
              &out) == 2);
    CHECK(out.find("--first-replica") != std::string::npos);
    CHECK(run({"--model", "voter", "--dims", "6x6", "--first-replica", "2"}, &out) == 2);   // task run
    // shown in the run summary
    CHECK(run(with(kBase, {"--replicas", "3", "--first-replica", "7"}), &out) == 0);
    CHECK(out.find("replicas 7..9 of the ensemble") != std::string::npos);
}
