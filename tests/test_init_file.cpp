#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "driver/Driver.hpp"

// Initial opinions from a file: --init file --init-path FILE [--init-fill random|x].

using namespace driver;
namespace fs = std::filesystem;

namespace {

struct DriverRun {
    int status = 0;
    std::string out, err;
};

DriverRun run(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* o = std::cout.rdbuf(out.rdbuf());
    std::streambuf* e = std::cerr.rdbuf(err.rdbuf());
    DriverRun r;
    r.status = run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    r.out = out.str();
    r.err = err.str();
    return r;
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// Temporary files with a common random stem, removed at the end.
struct TempFiles {
    std::string stem;
    TempFiles() {
        std::random_device rd;
        stem = (fs::temp_directory_path() / ("odsp_init_" + std::to_string(rd()))).string();
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

template <typename T>
Setup<T> setup_of(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    Options o = Options::from_args(static_cast<int>(argv.size()), argv.data());
    return read_setup<T>(o, 1);
}

template <typename T>
std::vector<T> start_of(const Setup<T>& s, unsigned seed) {
    std::mt19937 rng(seed);
    return s.factory(rng)->get_graph().states();
}

} // namespace

TEST_CASE(init_file_sets_lattice_states_exactly) {
    TempFiles tmp;
    // Indices 0..8 of a 3x3 lattice, in any order, with comments and mixed separators.
    const std::string f = tmp.write("_op.txt", "# node opinion\n4 -1\n0,1\n1;1\n2\t-1\n\n3 1\n% other comment\n"
                                               "5 -1\n6 1\n7 1\n8 -1\n");
    const auto s = setup_of<int>({"--model", "voter", "--dims", "3x3", "--init", "file", "--init-path", f});
    const std::vector<int> expected = {1, 1, -1, 1, -1, -1, 1, 1, -1};
    CHECK(start_of(s, 1) == expected);
    CHECK(start_of(s, 2) == expected);   // every replica starts from the file
    CHECK(contains(s.description, "opinions from " + f));
}

TEST_CASE(init_file_uses_edge_list_labels_after_the_largest_component) {
    TempFiles tmp;
    // The small component comes first, so the largest-component step renumbers
    // alice, bob, carol, dave from 2..5 to 0..3.
    const std::string net = tmp.write("_net.txt", "x y\nalice bob\nbob carol\ncarol dave\ndave alice\n");
    const std::string op = tmp.write("_op.txt", "dave 2\nalice 0\nx 1\nbob 1\ncarol 2\nghost 0\n");
    const auto s = setup_of<int>({"--model", "voter", "--graph", "file", "--path", net, "--init", "file", "--init-path", op});
    const std::map<std::string, int> expected = {{"alice", 0}, {"bob", 1}, {"carol", 2}, {"dave", 2}};
    const std::vector<std::string>& labels = s.topology->labels;
    CHECK(labels.size() == 4);
    const std::vector<int> st = start_of(s, 3);
    for (std::size_t i = 0; i < labels.size(); ++i) CHECK(st[i] == expected.at(labels[i]));
    CHECK(contains(s.description, "2 entries not in the graph"));   // x (dropped component), ghost (absent)
    std::mt19937 rng(1);
    CHECK(s.factory(rng)->number_of_opinions() == 3);   // multi-opinion start
}

TEST_CASE(init_file_missing_nodes_are_an_error_unless_filled) {
    TempFiles tmp;
    const std::string f = tmp.write("_op.txt", "0 1\n1 1\n2 -1\n3 1\n4 1\n5 1\n6 -1\n");   // 7 of 9
    DriverRun r = run({"--model", "voter", "--dims", "3x3", "--init", "file", "--init-path", f});
    CHECK(r.status == 2 && contains(r.err, "2 of the 9 nodes") && contains(r.err, "--init-fill"));

    const auto fixed = setup_of<int>({"--model", "voter", "--dims", "3x3", "--init", "file", "--init-path", f,
                                      "--init-fill", "-1"});
    CHECK(start_of(fixed, 1) == (std::vector<int>{1, 1, -1, 1, 1, 1, -1, -1, -1}));
    CHECK(contains(fixed.description, "2 filled with -1"));

    // Random fill draws from the file's own shares: here 9 of 10 given nodes hold 0.
    std::ostringstream text;
    for (int i = 0; i < 10; ++i) text << i * 10 << ' ' << (i == 3 ? 7 : 0) << '\n';
    const std::string g = tmp.write("_sparse.txt", text.str());
    const auto rnd = setup_of<int>({"--model", "voter", "--dims", "10x10", "--init", "file", "--init-path", g,
                                    "--init-fill", "random"});
    CHECK(contains(rnd.description, "90 filled at random"));
    int sevens = 0;
    std::set<int> seen;
    for (unsigned seed = 1; seed <= 5; ++seed) {
        const std::vector<int> st = start_of(rnd, seed);
        for (int i = 0; i < 10; ++i) CHECK(st[static_cast<std::size_t>(i * 10)] == (i == 3 ? 7 : 0));
        for (std::size_t k = 0; k < st.size(); ++k)
            if (k % 10 != 0) { seen.insert(st[k]); sevens += st[k] == 7; }
    }
    CHECK(seen == (std::set<int>{0, 7}));
    CHECK(sevens > 20 && sevens < 75);   // 450 draws with p = 0.1: mean 45, sd 6.4
    CHECK(start_of(rnd, 1) != start_of(rnd, 2));   // redrawn for every replica
}

TEST_CASE(init_file_continuous_opinions) {
    TempFiles tmp;
    const std::string f = tmp.write("_op.txt", "0 0.25\n1 0.75\n2 0.5\n");
    const auto s = setup_of<double>({"--model", "deffuant", "--epsilon", "0.3", "--dims", "4", "--init", "file",
                                     "--init-path", f, "--init-fill", "random"});
    const std::vector<double> st = start_of(s, 1);
    CHECK(st[0] == 0.25 && st[1] == 0.75 && st[2] == 0.5);
    CHECK(st[3] == 0.25 || st[3] == 0.75 || st[3] == 0.5);
}

TEST_CASE(init_file_with_zealots_on_top) {
    TempFiles tmp;
    std::ostringstream text;
    for (int i = 0; i < 16; ++i) text << i << ' ' << (i % 2 ? 1 : -1) << '\n';
    const std::string f = tmp.write("_op.txt", text.str());
    const auto s = setup_of<int>({"--model", "voter", "--dims", "4x4", "--init", "file", "--init-path", f,
                                  "--zealots", "3:1"});
    std::mt19937 rng(5);
    CHECK(s.factory(rng)->frozen_count() == 3);
}

TEST_CASE(init_file_errors) {
    TempFiles tmp;
    auto status_and = [](const DriverRun& r, const std::string& part) { return r.status == 2 && contains(r.err, part); };
    const std::vector<std::string> base = {"--model", "voter", "--dims", "4", "--init", "file", "--init-path"};
    auto with = [&](const std::string& path, std::vector<std::string> extra = {}) {
        std::vector<std::string> a = base;
        a.push_back(path);
        a.insert(a.end(), extra.begin(), extra.end());
        return run(a);
    };
    CHECK(status_and(with(tmp.write("_dup.txt", "0 1\n1 1\n0 -1\n2 1\n3 1\n")), "line 3: node 0 appears twice"));
    CHECK(status_and(with(tmp.write("_bad.txt", "0 1\n1 1\nx\n")), "line 3"));
    CHECK(status_and(with(tmp.write("_nan.txt", "0 1\n1 up\n")), "line 2: expected a number"));
    CHECK(status_and(with(tmp.write("_range.txt", "0 1\n1 1\n2 1\n4 1\n")), "line 4: node 4 is not an index 0..3"));
    CHECK(status_and(with(tmp.write("_frac.txt", "0 1\n1 0.5\n2 1\n3 1\n")), "integer"));
    CHECK(status_and(with(tmp.stem + "_does_not_exist.txt"), "cannot open"));
    CHECK(status_and(run({"--model", "voter", "--dims", "4", "--init", "file"}), "--init-path"));

    const std::string zero = tmp.write("_zero.txt", "0 1\n1 0\n2 1\n3 -1\n");
    CHECK(status_and(run({"--model", "ising", "--temperature", "2", "--dims", "4", "--init", "file", "--init-path", zero}),
                     "starts with opinion 0"));
    const std::string ok = tmp.write("_ok.txt", "0 1\n1 1\n");
    CHECK(status_and(run({"--model", "ising", "--temperature", "2", "--dims", "4", "--init", "file", "--init-path", ok,
                          "--init-fill", "0"}), "--init-fill"));
    CHECK(status_and(with(ok, {"--init-fill", "maybe"}), "--init-fill"));
    CHECK(status_and(run({"--model", "voter", "--graph", "gnp", "--n", "4", "--edge-prob", "0.9", "--init", "file",
                          "--init-path", ok, "--init-fill", "1", "--regenerate-graph", "true"}), "--regenerate-graph"));
    CHECK(status_and(run({"--model", "voter", "--dims", "4", "--init-fill", "1"}), "--init-fill"));   // unused
}

TEST_CASE(init_file_config_replays_exactly) {
    TempFiles tmp;
    std::ostringstream text;
    for (int i = 0; i < 36; i += 2) text << i << ' ' << (i % 3 ? 1 : -1) << '\n';
    const std::string f = tmp.write("_op.txt", text.str());
    CHECK(run({"--model", "voter", "--dims", "6x6", "--task", "ensemble", "--replicas", "6", "--init", "file", "--init-path",
               f, "--init-fill", "random", "--seed", "4", "--csv", tmp.stem + "_a"}).status == 0);
    CHECK(run({"--config", tmp.stem + "_a_config.txt", "--csv", tmp.stem + "_b"}).status == 0);
    const std::string a = read_without_comments(tmp.stem + "_a_replicas.csv");
    CHECK(!a.empty() && a == read_without_comments(tmp.stem + "_b_replicas.csv"));
}
