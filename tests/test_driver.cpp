#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "driver/Driver.hpp"

namespace fs = std::filesystem;
using namespace driver;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        std::random_device rd;
        path = fs::temp_directory_path() / ("odsp_driver_test_" + std::to_string(rd()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
    std::string file(const std::string& name) const { return (path / name).string(); }
};

// Runs the driver on the given arguments, capturing stdout and stderr.
struct DriverRun {
    int status = 0;
    std::string out, err;
};

DriverRun run(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* old_out = std::cout.rdbuf(out.rdbuf());
    std::streambuf* old_err = std::cerr.rdbuf(err.rdbuf());
    DriverRun r;
    r.status = run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

Options parse(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    return Options::from_args(static_cast<int>(argv.size()), argv.data());
}

std::string read_file(const std::string& path) { return read_without_comments(path); }   // data, without provenance

std::size_t lines(const std::string& text) { return static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n')); }

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

} // namespace

// ================================================================== Options

TEST_CASE(options_parse_command_line_forms) {
    Options o = parse({"--a", "1", "--b=two", "--flag", "--neg", "-0.5", "--last"});
    CHECK(o.integer("a") == 1);
    CHECK(o.str("b") == "two");
    CHECK(o.flag("flag", false));
    CHECK(o.num("neg") == -0.5);
    CHECK(o.flag("last", false));
    CHECK(o.num("absent", 2.5) == 2.5);
    CHECK_THROWS_AS(parse({"positional"}), OptionError);
}

TEST_CASE(options_typed_getters_report_bad_values) {
    Options o = parse({"--x", "abc", "--n", "1.5", "--b", "maybe", "--c", "blue"});
    CHECK_THROWS_AS(o.num("x"), OptionError);
    CHECK_THROWS_AS(o.integer("n"), OptionError);
    CHECK_THROWS_AS(o.flag("b", false), OptionError);
    CHECK_THROWS_AS(o.choice("c", "red", {"red", "green"}), OptionError);
    CHECK_THROWS_AS(o.str("missing"), OptionError);
    Options l = parse({"--f", "0.2, 0.3,0.5", "--w", "a, b,,c"});
    CHECK(l.num_list("f") == (std::vector<double>{0.2, 0.3, 0.5}));
    CHECK(l.word_list("w", "") == (std::vector<std::string>{"a", "b", "c"}));
}

TEST_CASE(options_config_file_and_override) {
    TempDir dir;
    std::ofstream(dir.file("run.cfg")) << "# comment\nmodel = ising   # trailing comment\n--temperature = 2.0\n\nverbose\n";
    Options o = parse({"--config", dir.file("run.cfg"), "--temperature", "2.5"});
    CHECK(o.str("model") == "ising");
    CHECK(o.num("temperature") == 2.5);   // command line wins
    CHECK(o.flag("verbose", false));
    CHECK(o.config_path() == dir.file("run.cfg"));
    std::ofstream(dir.file("bad.cfg")) << "model =\n";
    CHECK_THROWS_AS(parse({"--config", dir.file("bad.cfg")}), OptionError);
    CHECK_THROWS_AS(parse({"--config", dir.file("missing.cfg")}), OptionError);
}

TEST_CASE(options_track_use_and_effective_config) {
    Options o = parse({"--a", "1", "--typo", "2"});
    o.integer("a");
    o.num("b", 0.5);
    o.str("empty", "");
    CHECK(o.unused() == (std::vector<std::string>{"typo"}));
    CHECK(o.was_read("b") && !o.was_read("typo"));
    CHECK(o.effective_config() == "a = 1\nb = 0.5\n");   // defaults recorded, empty values left out
    Options t = parse({"--temprature", "2"});
    try {
        t.num("temperature");
        CHECK(false);
    } catch (const OptionError& e) {
        CHECK(contains(e.what(), "--temprature"));
    }
}

TEST_CASE(sweep_and_dims_parsing) {
    const SweepSpec list = parse_sweep("noise=0.1,0.2,0.3");
    CHECK(list.key == "noise" && list.values == (std::vector<std::string>{"0.1", "0.2", "0.3"}));
    const SweepSpec range = parse_sweep("p=0.05:0.1:0.01");   // inclusive, robust to rounding
    CHECK(range.values == (std::vector<std::string>{"0.05", "0.06", "0.07", "0.08", "0.09", "0.1"}));
    CHECK(parse_sweep("dynamics=metropolis,glauber").values.size() == 2);
    CHECK_THROWS_AS(parse_sweep("p"), OptionError);
    CHECK_THROWS_AS(parse_sweep("p=1:0:0.1"), OptionError);
    CHECK_THROWS_AS(parse_sweep("p=0:1:0"), OptionError);
    CHECK(parse_dims("4x5x6") == (std::vector<int>{4, 5, 6}));
    CHECK_THROWS_AS(parse_dims("4xx5"), OptionError);
    CHECK_THROWS_AS(parse_dims("0x5"), OptionError);
}

// ==================================================================== Setup

TEST_CASE(setup_builds_every_model_type) {
    const std::vector<std::vector<std::string>> configs = {
        {"--model", "voter", "--update", "link"},
        {"--model", "rf-voter"},
        {"--model", "noisy-voter", "--noise", "0.1", "--opinions", "3"},
        {"--model", "majority", "--tie", "keep"},
        {"--model", "qvoter", "--q", "3", "--p", "0.2", "--nonconformity", "anticonformity"},
        {"--model", "nonlinear-voter", "--alpha", "2"},
        {"--model", "majority-vote", "--noise", "0.1"},
        {"--model", "ising", "--temperature", "2", "--dynamics", "glauber"},
        {"--model", "potts", "--q", "4", "--temperature", "1"},
        {"--model", "sznajd"},
    };
    for (const auto& args : configs) {
        Options o = parse(args);
        o.set("dims", "6x6");
        Setup<int> s = read_setup<int>(o, 1);
        CHECK(o.unused().empty());
        std::mt19937 rng(1);
        auto m = s.factory(rng);
        CHECK(m->get_graph().size() == 36);
        const std::string& name = s.model.name;
        if (name == "voter") {
            const auto* v = dynamic_cast<VoterModel<int>*>(m.get());
            CHECK(v && v->update_scheme() == VoterUpdate::Link);
        }
        if (name == "rf-voter")        CHECK(dynamic_cast<RejectionFreeVoterModel<int>*>(m.get()) != nullptr);
        if (name == "noisy-voter")     CHECK(dynamic_cast<NoisyVoterModel<int>*>(m.get())->opinions().size() == 3);
        if (name == "majority")        CHECK(dynamic_cast<MajorityModel<int>*>(m.get())->tie_rule() == TieRule::KeepCurrent);
        if (name == "qvoter")          CHECK(dynamic_cast<QVoterModel<int>*>(m.get())->nonconformity() == Nonconformity::Anticonformity);
        if (name == "nonlinear-voter") CHECK(dynamic_cast<NonlinearVoterModel<int>*>(m.get())->alpha() == 2.0);
        if (name == "majority-vote")   CHECK(dynamic_cast<MajorityVoteModel<int>*>(m.get())->noise() == 0.1);
        if (name == "ising")           CHECK(dynamic_cast<IsingModel<int>*>(m.get())->dynamics() == IsingDynamics::Glauber);
        if (name == "potts")           CHECK(dynamic_cast<PottsModel<int>*>(m.get())->q() == 4);
        if (name == "sznajd")          CHECK(dynamic_cast<SznajdModel<int>*>(m.get()) != nullptr);
    }
    Options c = parse({"--model", "hk", "--epsilon", "0.2", "--synchronous", "false", "--graph", "complete", "--n", "50"});
    Setup<double> s = read_setup<double>(c, 1);
    std::mt19937 rng(1);
    auto m = s.factory(rng);
    CHECK(!dynamic_cast<HegselmannKrauseModel<double>*>(m.get())->synchronous());
    CHECK(m->get_graph().is_complete());
}

TEST_CASE(setup_builds_every_graph_type) {
    struct Case { std::vector<std::string> args; std::size_t nodes; };
    const std::vector<Case> cases = {
        {{"--graph", "lattice", "--dims", "4x5", "--boundary", "open"}, 20},
        {{"--graph", "complete", "--n", "30"}, 30},
        {{"--graph", "gnp", "--n", "200", "--edge-prob", "0.05"}, 200},
        {{"--graph", "gnm", "--n", "200", "--edges", "400"}, 200},
        {{"--graph", "regular", "--n", "100", "--k", "4"}, 100},
        {{"--graph", "ws", "--n", "100", "--k", "4", "--beta", "0.1"}, 100},
        {{"--graph", "ba", "--n", "100", "--m", "2"}, 100},
        {{"--graph", "star", "--n", "10"}, 10},
    };
    for (const auto& c : cases) {
        std::vector<std::string> args = {"--model", "voter"};
        args.insert(args.end(), c.args.begin(), c.args.end());
        Options o = parse(args);
        Setup<int> s = read_setup<int>(o, 2);
        CHECK(o.unused().empty());
        CHECK(s.topology->nodes == c.nodes);
    }
    // Regular graph: every degree is k.
    Options o = parse({"--model", "voter", "--graph", "regular", "--n", "50", "--k", "3", "--init", "up"});
    std::mt19937 rng(3);
    auto m = read_setup<int>(o, 4).factory(rng);
    for (std::size_t i = 0; i < 50; ++i) CHECK(m->get_graph().degree(i) == 3);
    CHECK(m->get_magnetization() == 50.0);
}

TEST_CASE(setup_graph_from_file_keeps_largest_component) {
    TempDir dir;
    std::ofstream(dir.file("net.txt")) << "a b\nb c\nc a\nx y\n";
    Options o = parse({"--model", "voter", "--graph", "file", "--path", dir.file("net.txt")});
    CHECK(read_setup<int>(o, 1).topology->nodes == 3);
    Options all = parse({"--model", "voter", "--graph", "file", "--path", dir.file("net.txt"), "--lcc", "false"});
    CHECK(read_setup<int>(all, 1).topology->nodes == 5);
}

TEST_CASE(setup_shares_or_regenerates_random_networks) {
    auto edges_of = [](Model<int>& m) {
        std::vector<std::size_t> v;
        for (std::size_t i = 0; i < m.get_graph().size(); ++i)
            for (std::size_t j : m.get_graph().neighbours(i)) v.push_back(i * 1000 + j);
        return v;
    };
    for (bool regenerate : {false, true}) {
        Options o = parse({"--model", "voter", "--graph", "ba", "--n", "100", "--m", "2",
                           "--regenerate-graph", regenerate ? "true" : "false"});
        Setup<int> s = read_setup<int>(o, 5);
        std::mt19937 r1(1), r2(2);
        auto a = s.factory(r1);
        auto b = s.factory(r2);
        CHECK((edges_of(*a) == edges_of(*b)) == !regenerate);
    }
}

TEST_CASE(setup_initial_conditions_and_zealots) {
    std::mt19937 rng(6);
    Options f = parse({"--model", "voter", "--graph", "complete", "--n", "100", "--init", "fractions", "--fractions", "0.2,0.8"});
    auto mf = read_setup<int>(f, 1).factory(rng);
    CHECK(mf->count_of(0) == 20 && mf->count_of(1) == 80);

    Options d = parse({"--model", "voter", "--graph", "complete", "--n", "40", "--init", "distinct"});
    CHECK(read_setup<int>(d, 1).factory(rng)->number_of_opinions() == 40);

    Options z = parse({"--model", "voter", "--graph", "ba", "--n", "200", "--m", "2", "--init", "down",
                       "--zealots", "5:1", "--zealot-placement", "hubs"});
    Setup<int> sz = read_setup<int>(z, 7);
    auto mz = sz.factory(rng);
    CHECK(mz->frozen_count() == 5 && mz->count_of(1) == 5);
    // The zealots are the five highest-degree nodes.
    std::vector<std::size_t> degrees;
    for (std::size_t i = 0; i < 200; ++i) degrees.push_back(mz->get_graph().degree(i));
    std::vector<std::size_t> sorted = degrees;
    std::sort(sorted.rbegin(), sorted.rend());
    for (std::size_t i = 0; i < 200; ++i)
        if (mz->is_frozen(i)) CHECK(degrees[i] >= sorted[4]);

    Options u = parse({"--model", "deffuant", "--epsilon", "0.2", "--graph", "complete", "--n", "100", "--lo", "2", "--hi", "3"});
    auto mu = read_setup<double>(u, 1).factory(rng);
    for (double x : mu->get_graph().states()) CHECK(x >= 2.0 && x < 3.0);

    CHECK_THROWS_AS([] { Options o = parse({"--model", "ising", "--temperature", "1", "--init", "distinct"});
                         read_setup<int>(o, 1); }(), OptionError);
    CHECK_THROWS_AS([] { Options o = parse({"--model", "voter", "--zealots", "3"}); read_setup<int>(o, 1); }(), OptionError);
}

TEST_CASE(observables_are_checked_against_the_model) {
    ModelSpec voter;  voter.name = "voter";
    ModelSpec ising;  ising.name = "ising";
    ModelSpec hk;     hk.name = "hk";
    CHECK_THROWS_AS(make_observable<int>("energy", voter), OptionError);
    CHECK_THROWS_AS(make_observable<int>("order", ising), OptionError);
    CHECK_THROWS_AS(make_observable<int>("clusters", voter), OptionError);
    CHECK_THROWS_AS(make_observable<double>("rho", hk), OptionError);
    CHECK_THROWS_AS(make_observable<double>("opinions", hk), OptionError);
    CHECK_THROWS_AS(make_observable<int>("magnetisation", voter), OptionError);
    try {
        make_observable<int>("rhoo", voter);
        CHECK(false);   // must throw
    } catch (const OptionError& e) {
        CHECK(contains(e.what(), "did you mean 'rho'"));
    }
    make_observable<int>("energy", ising);   // fine
    make_observable<double>("clusters", hk);
}

// =============================================================== end to end

TEST_CASE(driver_run_task_writes_time_series) {
    TempDir dir;
    const DriverRun r = run({"--model", "voter", "--dims", "8x8", "--steps", "100000", "--seed", "1",
                             "--csv", dir.file("v")});
    CHECK(r.status == 0);
    CHECK(contains(r.out, "Stopped after"));
    const std::string ts = read_file(dir.file("v_timeseries.csv"));
    CHECK(ts.rfind("sweep,m,rho\n0,", 0) == 0);
    CHECK(contains(ts, ",1,0\n") || contains(ts, ",-1,0\n"));   // ends in consensus: m = +-1, rho = 0
    CHECK(contains(read_file(dir.file("v_config.txt")), "seed = 1"));
}

TEST_CASE(driver_config_replay_is_exact) {
    TempDir dir;
    CHECK(run({"--model", "qvoter", "--q", "2", "--p", "0.05", "--graph", "ba", "--n", "300", "--m", "2",
               "--task", "ensemble", "--replicas", "20", "--steps", "200", "--csv", dir.file("a")}).status == 0);
    CHECK(run({"--config", dir.file("a_config.txt"), "--csv", dir.file("b")}).status == 0);
    CHECK(read_file(dir.file("a_replicas.csv")) == read_file(dir.file("b_replicas.csv")));
    CHECK(lines(read_file(dir.file("a_replicas.csv"))) == 21);
}

TEST_CASE(driver_ensemble_and_stationary_summaries) {
    TempDir dir;
    const DriverRun e = run({"--model", "voter", "--graph", "complete", "--n", "50", "--init", "fractions",
                             "--fractions", "0.3,0.7", "--task", "ensemble", "--replicas", "200", "--seed", "2",
                             "--csv", dir.file("e")});
    CHECK(e.status == 0);
    CHECK(contains(e.out, "Consensus reached: 200 (100%)"));
    const std::string fix = read_file(dir.file("e_fixation.csv"));
    CHECK(fix.rfind("opinion,replicas,probability\n0,", 0) == 0);

    const DriverRun s = run({"--model", "ising", "--temperature", "1.5", "--init", "up", "--dims", "8x8",
                             "--task", "stationary", "--samples", "500", "--burn-in", "100", "--seed", "3",
                             "--csv", dir.file("s")});
    CHECK(s.status == 0);
    CHECK(contains(s.out, "Binder cumulant"));
    CHECK(lines(read_file(dir.file("s_samples.csv"))) == 501);
}

TEST_CASE(driver_sweep_writes_one_row_per_value) {
    TempDir dir;
    const DriverRun r = run({"--model", "majority-vote", "--init", "up", "--dims", "8x8", "--task", "stationary",
                             "--samples", "200", "--burn-in", "50", "--sweep", "noise=0.1:0.3:0.1", "--seed", "4",
                             "--csv", dir.file("sw")});
    CHECK(r.status == 0);
    const std::string table = read_file(dir.file("sw_sweep.csv"));
    CHECK(lines(table) == 4);
    CHECK(table.rfind("noise,mean,mean_err,", 0) == 0);
    CHECK(contains(table, "\n0.1,") && contains(table, "\n0.2,") && contains(table, "\n0.3,"));
    // Sweeping an option the configuration does not use is an error.
    const DriverRun bad = run({"--model", "voter", "--task", "stationary", "--sweep", "temperature=1,2"});
    CHECK(bad.status == 2);
}

TEST_CASE(driver_reports_errors_with_exit_codes) {
    CHECK(run({"--model", "voter", "--temprature", "2"}).status == 2);
    const DriverRun unused = run({"--model", "voter", "--temperature", "2"});
    CHECK(unused.status == 2 && contains(unused.err, "--temperature does not apply"));
    const DriverRun typo = run({"--model", "voter", "--stpes", "10"});
    CHECK(typo.status == 2 && contains(typo.err, "did you mean --steps"));
    const DriverRun model = run({"--model", "votr"});
    CHECK(model.status == 2 && contains(model.err, "did you mean 'voter'"));
    CHECK(run({"--model", "noisy-voter", "--noise", "2"}).status == 2);                // rejected by the model
    CHECK(run({"--model", "voter", "--graph", "file", "--path", "/no/such/file"}).status == 1);
    CHECK(run({"--model", "voter", "--task", "run", "--sweep", "up=0.1,0.2"}).status == 2);
    const DriverRun help = run({"--help", "models"});
    CHECK(help.status == 0 && contains(help.out, "qvoter"));
    CHECK(run({}).status == 0);
}
