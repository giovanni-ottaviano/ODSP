#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "driver/Driver.hpp"

// --save-states: final configurations to prefix_states.csv, in the format
// --init file reads back.

using namespace driver;
namespace fs = std::filesystem;

namespace {

int run(std::vector<std::string> args) {
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

struct TempFiles {
    std::string stem;
    TempFiles() {
        std::random_device rd;
        stem = (fs::temp_directory_path() / ("odsp_states_" + std::to_string(rd()))).string();
    }
    ~TempFiles() {
        std::error_code ec;
        const fs::path dir = fs::path(stem).parent_path();
        const std::string name = fs::path(stem).filename().string();
        for (const auto& f : fs::directory_iterator(dir, ec))
            if (f.path().filename().string().rfind(name, 0) == 0) fs::remove(f.path(), ec);
    }
    std::string write(const std::string& suffix, const std::string& text) const {
        std::ofstream(stem + suffix) << text;
        return stem + suffix;
    }
};

// A states file: the commented header's column names, and node -> values.
struct States {
    std::vector<std::string> columns;   // after "# node"
    std::vector<std::string> nodes;
    std::vector<std::vector<double>> values;   // values[column][row]
};

States read_states(const std::string& path) {
    States s;
    std::ifstream in(path);
    std::string line;
    bool header = false;
    while (std::getline(in, line)) {
        if (line.rfind("# node", 0) == 0) {   // the header is the last comment line
            std::stringstream hs(line.substr(2));
            std::string c;
            std::getline(hs, c, ',');
            while (std::getline(hs, c, ',')) s.columns.push_back(c);
            s.values.assign(s.columns.size(), {});
            header = true;
            continue;
        }
        if (line.empty() || line[0] == '#') continue;
        CHECK(header);
        std::stringstream ls(line);
        std::string cell;
        std::getline(ls, cell, ',');
        s.nodes.push_back(cell);
        for (std::size_t k = 0; std::getline(ls, cell, ','); ++k) s.values[k].push_back(std::stod(cell));
    }
    return s;
}

std::map<std::string, std::vector<double>> read_csv(const std::string& path) {
    std::istringstream in(read_without_comments(path));
    std::string line, cell;
    std::getline(in, line);
    std::vector<std::string> names;
    std::stringstream hs(line);
    while (std::getline(hs, cell, ',')) names.push_back(cell);
    std::map<std::string, std::vector<double>> cols;
    while (std::getline(in, line)) {
        std::stringstream ls(line);
        for (std::size_t k = 0; std::getline(ls, cell, ','); ++k) cols[names[k]].push_back(std::stod(cell));
    }
    return cols;
}

double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

template <typename T>
std::vector<T> start_from(const std::string& states, std::vector<std::string> graph) {
    std::vector<std::string> args = {"odsp"};
    args.insert(args.end(), graph.begin(), graph.end());
    args.insert(args.end(), {"--init", "file", "--init-path", states});
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    Options o = Options::from_args(static_cast<int>(argv.size()), argv.data());
    const Setup<T> s = read_setup<T>(o, 1);
    std::mt19937 rng(1);
    return s.factory(rng)->get_graph().states();
}

} // namespace

TEST_CASE(save_states_run_final_configuration_loads_back) {
    TempFiles tmp;
    CHECK(run({"--model", "voter", "--dims", "8x8", "--steps", "7", "--stop", "false", "--observables", "m", "--seed", "3",
               "--save-states", "true", "--csv", tmp.stem}) == 0);
    const States st = read_states(tmp.stem + "_states.csv");
    CHECK(st.columns == std::vector<std::string>{"opinion"});
    CHECK(st.nodes.size() == 64 && st.nodes.front() == "0" && st.nodes.back() == "63");
    const auto ts = read_csv(tmp.stem + "_timeseries.csv");
    CHECK(std::abs(mean(st.values[0]) - ts.at("m").back()) < 1e-12);   // the state at the end
    const std::vector<int> loaded = start_from<int>(tmp.stem + "_states.csv", {"--model", "voter", "--dims", "8x8"});
    for (std::size_t i = 0; i < 64; ++i) CHECK(loaded[i] == static_cast<int>(st.values[0][i]));
}

TEST_CASE(save_states_ensemble_one_column_per_replica) {
    TempFiles a, b;
    for (auto* t : {&a, &b})
        CHECK(run({"--model", "voter", "--dims", "6x6", "--task", "ensemble", "--replicas", "5", "--steps", "20", "--seed", "4",
                   "--threads", t == &a ? "1" : "4", "--save-states", "true", "--csv", t->stem}) == 0);
    const States st = read_states(a.stem + "_states.csv");
    CHECK(st.columns.size() == 5 && st.columns[0] == "replica0" && st.columns[4] == "replica4");
    const auto rep = read_csv(a.stem + "_replicas.csv");
    for (std::size_t r = 0; r < 5; ++r) CHECK(std::abs(mean(st.values[r]) - rep.at("m")[r]) < 1e-12);
    CHECK(read_without_comments(a.stem + "_states.csv") == read_without_comments(b.stem + "_states.csv"));
}

TEST_CASE(save_states_stationary_one_column_per_chain) {
    TempFiles tmp;
    CHECK(run({"--model", "ising", "--temperature", "3", "--dims", "8x8", "--task", "stationary", "--replicas", "3",
               "--burn-in", "10", "--samples", "20", "--seed", "5", "--save-states", "true", "--csv", tmp.stem}) == 0);
    const States st = read_states(tmp.stem + "_states.csv");
    CHECK(st.columns == (std::vector<std::string>{"chain0", "chain1", "chain2"}));
    CHECK(st.nodes.size() == 64);
    for (const auto& col : st.values)
        for (double v : col) CHECK(v == 1.0 || v == -1.0);
}

TEST_CASE(save_states_keyed_by_edge_list_labels) {
    TempFiles tmp;
    const std::string net = tmp.write("_net.txt", "x y\nalice bob\nbob carol\ncarol dave\ndave alice\nalice carol\n");
    const std::vector<std::string> graph = {"--model", "voter", "--graph", "file", "--path", net};
    std::vector<std::string> args = graph;
    args.insert(args.end(), {"--init", "fractions", "--fractions", "0.5,0.5", "--steps", "1", "--stop", "false", "--seed",
                             "6", "--save-states", "true", "--csv", tmp.stem + "_r"});
    CHECK(run(args) == 0);
    const States st = read_states(tmp.stem + "_r_states.csv");
    CHECK(st.nodes.size() == 4);
    for (const auto& n : st.nodes) CHECK(n == "alice" || n == "bob" || n == "carol" || n == "dave");
    // Loading it back gives each labelled node its saved opinion.
    std::map<std::string, double> saved;
    for (std::size_t i = 0; i < st.nodes.size(); ++i) saved[st.nodes[i]] = st.values[0][i];
    std::vector<std::string> a = {"odsp"};
    a.insert(a.end(), graph.begin(), graph.end());
    a.insert(a.end(), {"--init", "file", "--init-path", tmp.stem + "_r_states.csv"});
    std::vector<char*> argv;
    for (auto& s : a) argv.push_back(s.data());
    Options o = Options::from_args(static_cast<int>(argv.size()), argv.data());
    const Setup<int> s = read_setup<int>(o, 1);
    std::mt19937 rng(1);
    const std::vector<int> loaded = s.factory(rng)->get_graph().states();
    for (std::size_t i = 0; i < loaded.size(); ++i) CHECK(loaded[i] == saved.at(s.topology->labels[i]));
}

TEST_CASE(save_states_continuous_opinions_round_trip_exactly) {
    TempFiles tmp;
    CHECK(run({"--model", "deffuant", "--epsilon", "0.3", "--dims", "5x5", "--steps", "3", "--stop", "false", "--seed", "7",
               "--save-states", "true", "--csv", tmp.stem}) == 0);
    const States st = read_states(tmp.stem + "_states.csv");
    const std::vector<double> loaded =
        start_from<double>(tmp.stem + "_states.csv", {"--model", "deffuant", "--epsilon", "0.3", "--dims", "5x5"});
    CHECK(loaded == st.values[0]);   // bit for bit
}

TEST_CASE(save_states_in_sweeps_and_errors) {
    TempFiles tmp;
    CHECK(run({"--model", "voter", "--graph", "complete", "--task", "ensemble", "--replicas", "3", "--sweep", "n=10,20",
               "--seed", "8", "--save-states", "true", "--csv", tmp.stem + "_e"}) == 0);
    CHECK(read_states(tmp.stem + "_e_point0_states.csv").nodes.size() == 10);
    CHECK(read_states(tmp.stem + "_e_point1_states.csv").nodes.size() == 20);
    CHECK(run({"--model", "ising", "--dims", "6x6", "--task", "stationary", "--samples", "10", "--burn-in", "5", "--sweep",
               "temperature=2,3", "--seed", "9", "--save-states", "true", "--csv", tmp.stem + "_s"}) == 0);
    CHECK(read_states(tmp.stem + "_s_point1_states.csv").columns == std::vector<std::string>{"chain0"});
    CHECK(!fs::exists(tmp.stem + "_s_point0_samples.csv"));   // only the states, not the samples

    CHECK(run({"--model", "voter", "--dims", "6x6", "--steps", "2", "--save-states", "true"}) == 2);   // needs --csv
    CHECK(run({"--model", "voter", "--dims", "6x6", "--steps", "2", "--save-states", "true", "--provenance", "false",
               "--csv", tmp.stem + "_p"}) == 0);
    std::ifstream in(tmp.stem + "_p_states.csv");
    std::string first;
    std::getline(in, first);
    CHECK(first == "# node,opinion");
}
