#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "TestFramework.hpp"

// Runs every registered test case, or only those whose name contains the
// optional command-line filter:  bin/tests [substring]
int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";

    // Ensembles with threads = 0 ("all cores") use at most 8 threads in the
    // tests, unless the environment already sets a lower cap.
    const char* cap = std::getenv("ODSP_MAX_THREADS");
    const int current = cap ? std::atoi(cap) : 0;
    if (current < 1 || current > 8) setenv("ODSP_MAX_THREADS", "8", 1);

    int run = 0, failed = 0;
    for (const odsp_test::TestCase& t : odsp_test::registry()) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) continue;
        ++run;

        const auto start = std::chrono::steady_clock::now();
        std::string error;
        try {
            t.fn();
        } catch (const odsp_test::Failure& f) {
            error = f.message;
        } catch (const std::exception& e) {
            error = std::string(t.file) + ": unexpected exception: " + e.what();
        } catch (...) {
            error = std::string(t.file) + ": unexpected non-standard exception";
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start).count();

        if (error.empty()) {
            std::cout << "[ PASS ] " << t.name << "  (" << static_cast<long long>(ms) << " ms)\n";
        } else {
            ++failed;
            std::cout << "[ FAIL ] " << t.name << "\n         " << error << '\n';
        }
    }

    std::cout << "\n" << run - failed << " / " << run << " test cases passed, "
              << odsp_test::check_count() << " checks.\n";
    if (run == 0) {
        std::cout << "No test case matches '" << filter << "'.\n";
        return 1;
    }
    return failed == 0 ? 0 : 1;
}
