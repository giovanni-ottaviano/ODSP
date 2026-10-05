#ifndef ODSP_DRIVER_OPTIONS_HPP
#define ODSP_DRIVER_OPTIONS_HPP

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Key/value options for the odsp driver, from the command line and/or a config file:
//
//   odsp --model ising --temperature 2.2 --graph lattice --dims 32x32
//   odsp --config run.cfg --temperature 2.3        (command line overrides the file)
//
// Command line: "--key value", "--key=value", or a bare "--flag" (= true) when
// the next token also starts with "--". Config file: "key = value" lines
// ('#' starts a comment; a bare "key" line means true).
//
// Every getter records the value it returned, including defaults, so the
// complete EFFECTIVE configuration can be printed and saved for exact reruns,
// and options that were given but never read can be reported as errors

namespace driver {

class OptionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Options {
public:
    // --- building ---
    void set(const std::string& key, const std::string& value) { _given[key] = value; }

    static Options from_args(int argc, char** argv) {
        Options file_opts, cli;
        std::string config;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--", 0) != 0 || arg.size() == 2)
                throw OptionError("unexpected argument '" + arg + "' (options look like --key value)");

            arg = arg.substr(2);
            std::string key = arg, value;
            const std::size_t eq = arg.find('=');
            if (eq != std::string::npos) {
                key = arg.substr(0, eq);
                value = arg.substr(eq + 1);
            } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                value = argv[++i];
            } else {
                value = "true";
            }
            if (key == "config") config = value;
            else cli.set(key, value);
        }
        if (!config.empty())
            file_opts = from_file(config);
        for (const auto& kv : cli._given)
            file_opts.set(kv.first, kv.second);

        file_opts._config_path = config;
        return file_opts;
    }

    static Options from_file(const std::string& path) {
        std::ifstream in(path);
        if (!in)
            throw OptionError("cannot open config file '" + path + "'");
        Options opts;
        std::string line;
        int line_no = 0;

        while (std::getline(in, line)) {
            ++line_no;
            const std::size_t hash = line.find('#');
            if (hash != std::string::npos) line.erase(hash);
            const std::size_t eq = line.find('=');
            std::string key = _trim(eq == std::string::npos ? line : line.substr(0, eq));
            std::string value = eq == std::string::npos ? "true" : _trim(line.substr(eq + 1));
            if (key.empty()) continue;
            if (key.rfind("--", 0) == 0) key = key.substr(2);
            if (eq != std::string::npos && value.empty())
                throw OptionError(path + ":" + std::to_string(line_no) + ": option '" + key + "' has no value");

            opts.set(key, value);
        }
        return opts;
    }

    // --- reading (each call records the effective value) ---
    bool has(const std::string& key) const { return _given.count(key) > 0; }

    std::string str(const std::string& key, const std::string& fallback) { return _use(key, fallback, true); }
    std::string str(const std::string& key) { return _use(key, "", false); }

    double num(const std::string& key, double fallback) { return _to_double(key, _use(key, _fmt(fallback), true)); }
    double num(const std::string& key) { return _to_double(key, _use(key, "", false)); }

    long long integer(const std::string& key, long long fallback) {
        return _to_integer(key, _use(key, std::to_string(fallback), true));
    }
    long long integer(const std::string& key) { return _to_integer(key, _use(key, "", false)); }

    bool flag(const std::string& key, bool fallback) {
        const std::string v = _use(key, fallback ? "true" : "false", true);
        if (v == "true" || v == "yes" || v == "1" || v == "on") return true;
        if (v == "false" || v == "no" || v == "0" || v == "off") return false;
        throw OptionError("option --" + key + ": expected true/false, got '" + v + "'");
    }

    // One of the allowed words.
    std::string choice(const std::string& key, const std::string& fallback, const std::vector<std::string>& allowed) {
        const std::string v = str(key, fallback);
        if (std::find(allowed.begin(), allowed.end(), v) == allowed.end()) {
            std::string list;
            for (const auto& a : allowed) list += (list.empty() ? "" : ", ") + a;
            throw OptionError("option --" + key + ": '" + v + "' is not one of: " + list);
        }
        return v;
    }

    // Comma-separated list of numbers, e.g. "0.2,0.3,0.5".
    std::vector<double> num_list(const std::string& key) {
        const std::string v = str(key);
        std::vector<double> out;
        std::stringstream ss(v);
        std::string item;

        while (std::getline(ss, item, ','))
            out.push_back(_to_double(key, _trim(item)));
        if (out.empty())
            throw OptionError("option --" + key + ": empty list");

        return out;
    }

    // Comma-separated words.
    std::vector<std::string> word_list(const std::string& key, const std::string& fallback) {
        const std::string v = str(key, fallback);
        std::vector<std::string> out;
        std::stringstream ss(v);
        std::string item;

        while (std::getline(ss, item, ',')) {
            item = _trim(item);
            if (!item.empty()) out.push_back(item);
        }
        return out;
    }

    // --- checking and reporting ---
    bool was_read(const std::string& key) const { return _effective.count(key) > 0; }

    // Options that were given but never read.
    std::vector<std::string> unused() const {
        std::vector<std::string> out;
        for (const auto& kv : _given)
            if (!_effective.count(kv.first)) out.push_back(kv.first);

        return out;
    }

    // key = value for everything read, in order of first use
    std::string effective_config() const {
        std::string out;
        for (const std::string& key : _order)
            if (!_effective.at(key).empty())
                out += key + " = " + _effective.at(key) + "\n";

        return out;
    }

    const std::string& config_path() const { return _config_path; }

private:
    std::string _use(const std::string& key, const std::string& fallback, bool has_fallback) {
        std::string value;
        const auto it = _given.find(key);
        if (it != _given.end())
            value = it->second;
        else if (has_fallback)
            value = fallback;
        else
            throw OptionError("missing required option --" + key + _typo_hint(key));

        if (!_effective.count(key))
            _order.push_back(key);

        _effective[key] = value;
        return value;
    }

    // if a given, unread option is close to key
    std::string _typo_hint(const std::string& key) const;

    static double _to_double(const std::string& key, const std::string& v) {
        try {
            std::size_t used = 0;
            const double d = std::stod(v, &used);
            if (used == v.size()) return d;
        } catch (const std::exception&) {}
        throw OptionError("option --" + key + ": expected a number, got '" + v + "'");
    }

    static long long _to_integer(const std::string& key, const std::string& v) {
        try {
            std::size_t used = 0;
            const long long n = std::stoll(v, &used);
            if (used == v.size()) return n;
        } catch (const std::exception&) {}
        throw OptionError("option --" + key + ": expected an integer, got '" + v + "'");
    }

    static std::string _fmt(double d) {
        std::ostringstream os;
        os.precision(17);
        os << d;

        return os.str();
    }

    static std::string _trim(const std::string& s) {
        const std::size_t a = s.find_first_not_of(" \t\r");
        if (a == std::string::npos) return "";
        const std::size_t b = s.find_last_not_of(" \t\r");

        return s.substr(a, b - a + 1);
    }

    std::map<std::string, std::string> _given;       // as provided
    std::map<std::string, std::string> _effective;   // as read (with defaults)
    std::vector<std::string>           _order;       // keys in order of first read
    std::string                        _config_path;
};

inline std::size_t edit_distance(const std::string& a, const std::string& b);

inline std::string Options::_typo_hint(const std::string& key) const {
    for (const auto& kv : _given)
        if (!_effective.count(kv.first) && edit_distance(kv.first, key) <= 2)
            return " (you gave --" + kv.first + ": a typo?)";
    return "";
}

// Edit distance, for "did you mean" suggestions.
inline std::size_t edit_distance(const std::string& a, const std::string& b) {
    std::vector<std::size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j)
        prev[j] = j;
    for (std::size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});

        prev.swap(cur);
    }
    return prev[b.size()];
}

} // namespace driver

#endif // ODSP_DRIVER_OPTIONS_HPP
