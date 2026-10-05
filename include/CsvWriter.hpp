#ifndef ODSP_CSVWRITER_HPP
#define ODSP_CSVWRITER_HPP

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Tiny CSV writer for simulation output: optional '#' comment lines (the
// preamble, see set_preamble), one header row, then rows of values. Readers
// skip the comments with pandas.read_csv(f, comment="#") or
// numpy.genfromtxt(f, comments="#", names=True).
// Floating-point values are written in the shortest form that reads back to
// exactly the same value (0.3 stays "0.3"), so results can be re-analysed
// without loss. Fields are not quoted, so they must not contain
// commas or newlines (true for all numeric and label data written here).
//
// An empty path gives a DISABLED writer whose row() does nothing, so callers
// can write unconditionally and let the command line decide whether a file
// is produced:
//
//   CsvWriter out(opt.csv_path("rho"), {"model", "sweep", "rho"});
//   out.row("voter", 16, 0.123);
class CsvWriter {
public:
    CsvWriter(const std::string& path, std::initializer_list<const char*> header)
        : CsvWriter(path, std::vector<std::string>(header.begin(), header.end())) {}

    // Header known only at run time (e.g. user-selected columns).
    CsvWriter(const std::string& path, const std::vector<std::string>& header) : _path(path) {
        if (_path.empty()) return;

        const std::filesystem::path parent = std::filesystem::path(_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);

        _out.open(_path);
        if (!_out)
            throw std::runtime_error("CsvWriter: cannot open '" + _path + "' for writing.");

        const std::string& pre = _preamble_storage();
        if (!pre.empty()) {
            std::size_t start = 0;
            while (start <= pre.size()) {
                const std::size_t end = std::min(pre.find('\n', start), pre.size());
                const std::string line = pre.substr(start, end - start);
                _out << (line.empty() ? "#" : "# " + line) << '\n';
                start = end + 1;
            }
        }

        bool first = true;
        for (const std::string& h : header) {
            _out << (first ? "" : ",") << h;
            first = false;
        }
        _out << '\n';
    }

    // A row whose fields are only known at run time: a label then numbers.
    void row_values(const std::string& label, const std::vector<double>& values) {
        if (!enabled()) return;
        _out << label;
        for (double v : values) { _out << ','; _field(v); }
        _out << '\n';
    }

    template <typename... Fields>
    void row(const Fields&... fields) {
        if (!enabled()) return;
        bool first = true;
        ((_out << (first ? "" : ","), _field(fields), first = false), ...);
        _out << '\n';
    }

    // Text written before the header of every file opened afterwards, each line
    // prefixed with "# " (provenance: version, command, configuration, ...).
    // Set it once, before any threads write files; "" turns it off.
    static void set_preamble(const std::string& text) { _preamble_storage() = text; }
    static std::string preamble() { return _preamble_storage(); }

    bool               enabled() const { return !_path.empty(); }
    const std::string& path()    const { return _path; }

private:
    static std::string& _preamble_storage() {
        static std::string text;
        return text;
    }

    template <typename F>
    void _field(const F& value) {
        if constexpr (std::is_floating_point_v<F>) {
            char buf[64];
            const auto res = std::to_chars(buf, buf + sizeof(buf), value);   // shortest round-trip
            _out.write(buf, res.ptr - buf);
        } else {
            _out << value;
        }
    }

    std::string   _path;
    std::ofstream _out;
};

#endif // ODSP_CSVWRITER_HPP
