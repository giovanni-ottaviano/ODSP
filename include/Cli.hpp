#ifndef ODSP_CLI_HPP
#define ODSP_CLI_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "CsvWriter.hpp"
#include "Version.hpp"

// Minimal command-line handling shared by the demo programs.
//
//   --seed <n>       seed the master RNG (default: drawn from std::random_device)
//   --csv <prefix>   also write the results as CSV files named <prefix>_*.csv;
//                    the prefix may contain a directory, which is created
//   --threads <n>    worker threads for ensemble runs (default 0 = all cores);
//                    results do not depend on it
//   --help           print usage and exit
//
// A program can declare extra options that take one value (see CliExtraOption);
// their values land in CliOptions::extra.
//
// The seed actually used is always printed, so any run can be reproduced
// exactly by passing it back with --seed; it is also recorded, with the ODSP
// version, commit and command line, at the top of every CSV file written.
struct CliExtraOption {
    std::string name;         // e.g. "--edges"
    std::string value_name;   // e.g. "<file>", shown in the usage line
    std::string help;
};

struct CliOptions {
    unsigned    seed       = 0;
    bool        seed_given = false;
    std::string csv_prefix;          // empty => no CSV output
    int         threads    = 0;      // 0 => all hardware threads
    std::map<std::string, std::string> extra;   // values of the program's extra options

    // Value of an extra option, or "" if it was not given.
    std::string get(const std::string& name) const {
        const auto it = extra.find(name);
        return it == extra.end() ? std::string() : it->second;
    }

    // Path of the CSV file `name`, or "" when CSV output is off (which gives a
    // disabled CsvWriter).
    bool        csv_enabled() const { return !csv_prefix.empty(); }
    std::string csv_path(const std::string& name) const {
        return csv_enabled() ? csv_prefix + "_" + name + ".csv" : std::string();
    }
};

inline void print_usage(const char* program, const char* description,
                        const std::vector<CliExtraOption>& extra_options = {}) {
    std::cout << description << "\n\n"
              << "Usage: " << program << " [--seed <n>] [--csv <prefix>] [--threads <n>]";
    for (const auto& o : extra_options) std::cout << " [" << o.name << ' ' << o.value_name << ']';
    std::cout << " [--help]\n"
              << "  --seed <n>       seed the master RNG (default: random)\n"
              << "  --csv <prefix>   also write results to <prefix>_*.csv\n"
              << "  --threads <n>    threads for ensemble runs (default 0 = all cores,\n"
              << "                   at most $ODSP_MAX_THREADS if set)\n";
    for (const auto& o : extra_options) {
        std::string flag = o.name + ' ' + o.value_name;
        flag.resize(std::max<std::size_t>(flag.size() + 1, 17), ' ');
        std::cout << "  " << flag << o.help << '\n';
    }
    std::cout << "  --help           show this message\n";
}

// Parses argv; on --help prints usage and exits 0, on bad input prints the
// error plus usage and exits 2. Without --seed, *default_seed is used if given,
// otherwise a seed is drawn from std::random_device.
inline CliOptions parse_cli(int argc, char** argv, const char* description,
                            const unsigned* default_seed = nullptr,
                            const std::vector<CliExtraOption>& extra_options = {}) {
    CliOptions opt;
    const char* program = argc > 0 ? argv[0] : "odsp";

    auto fail = [&](const std::string& msg) {
        std::cerr << program << ": " << msg << "\n\n";
        print_usage(program, description, extra_options);
        std::exit(2);
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(program, description, extra_options);
            std::exit(0);
        } else if (arg == "--seed") {
            if (i + 1 >= argc) fail("--seed needs a value");
            const std::string value = argv[++i];
            try {
                std::size_t used = 0;
                const unsigned long v = std::stoul(value, &used);
                if (used != value.size() || v > 0xFFFFFFFFUL) throw std::out_of_range(value);
                opt.seed = static_cast<unsigned>(v);
            } catch (const std::exception&) {
                fail("invalid seed '" + value + "' (expected an unsigned 32-bit integer)");
            }
            opt.seed_given = true;
        } else if (arg == "--csv") {
            if (i + 1 >= argc) fail("--csv needs a value");
            opt.csv_prefix = argv[++i];
            if (opt.csv_prefix.empty()) fail("--csv prefix must not be empty");
        } else if (arg == "--threads") {
            if (i + 1 >= argc) fail("--threads needs a value");
            const std::string value = argv[++i];
            try {
                std::size_t used = 0;
                opt.threads = std::stoi(value, &used);
                if (used != value.size() || opt.threads < 0) throw std::out_of_range(value);
            } catch (const std::exception&) {
                fail("invalid thread count '" + value + "' (expected an integer >= 0)");
            }
        } else {
            const auto it = std::find_if(extra_options.begin(), extra_options.end(),
                                         [&arg](const CliExtraOption& o) { return o.name == arg; });
            if (it == extra_options.end()) fail("unknown argument '" + arg + "'");
            if (i + 1 >= argc) fail(arg + " needs a value");
            opt.extra[arg] = argv[++i];
        }
    }

    if (!opt.seed_given)
        opt.seed = default_seed ? *default_seed : std::random_device{}();

    std::cout << "seed = " << opt.seed << "   (rerun with --seed " << opt.seed << ")\n\n";

    // Provenance at the top of every CSV the program writes.
    std::string command = program;
    for (int i = 1; i < argc; ++i) command += std::string(" ") + argv[i];
    CsvWriter::set_preamble(odsp_version_string() + ", written " + iso_timestamp() + "\ncommand: " + command +
                            "\nseed = " + std::to_string(opt.seed));
    return opt;
}

#endif // ODSP_CLI_HPP
