#ifndef ODSP_TESTFRAMEWORK_HPP
#define ODSP_TESTFRAMEWORK_HPP

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

// Minimal self-contained test harness (no external dependency).
//
//   TEST_CASE(lattice_has_2d_neighbours) {
//       CHECK(lat.neighbours(0).size() == 4);
//       CHECK_NEAR(p_up, 0.3, 0.04);
//       CHECK_THROWS_AS(RegularLattice<int>({1, 1}, {2}), std::invalid_argument);
//   }
//
// Test cases self-register at static-initialisation time and are run by
// tests/test_main.cpp. The first failing check aborts its test case (by
// throwing odsp_test::Failure) and the run continues with the next one.
namespace odsp_test {

struct Failure {
    std::string message;
};

struct TestCase {
    const char* name;
    const char* file;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline long long& check_count() {
    static long long n = 0;
    return n;
}

struct Registrar {
    Registrar(const char* name, const char* file, void (*fn)()) {
        registry().push_back({name, file, fn});
    }
};

[[noreturn]] inline void fail(const char* file, int line, const std::string& what) {
    std::ostringstream os;
    os << file << ":" << line << ": " << what;
    throw Failure{os.str()};
}

} // namespace odsp_test

#define TEST_CASE(name)                                                         \
    static void name();                                                         \
    static const odsp_test::Registrar name##_registrar(#name, __FILE__, &name); \
    static void name()

#define CHECK(cond)                                                             \
    do {                                                                        \
        ++odsp_test::check_count();                                             \
        if (!(cond)) odsp_test::fail(__FILE__, __LINE__, "CHECK(" #cond ") failed"); \
    } while (0)

// |actual - expected| <= tol, reporting both values on failure.
#define CHECK_NEAR(actual, expected, tol)                                       \
    do {                                                                        \
        ++odsp_test::check_count();                                             \
        const double a_ = static_cast<double>(actual);                          \
        const double e_ = static_cast<double>(expected);                        \
        const double t_ = static_cast<double>(tol);                             \
        if (!(std::fabs(a_ - e_) <= t_)) {                                      \
            std::ostringstream os_;                                             \
            os_ << "CHECK_NEAR(" #actual ", " #expected ", " #tol ") failed: "  \
                << a_ << " vs " << e_ << " (|diff| = " << std::fabs(a_ - e_)    \
                << " > " << t_ << ")";                                          \
            odsp_test::fail(__FILE__, __LINE__, os_.str());                     \
        }                                                                       \
    } while (0)

#define CHECK_THROWS_AS(expr, exception_type)                                   \
    do {                                                                        \
        ++odsp_test::check_count();                                             \
        bool thrown_ = false;                                                   \
        try { (void)(expr); }                                                   \
        catch (const exception_type&) { thrown_ = true; }                       \
        catch (...) {                                                           \
            odsp_test::fail(__FILE__, __LINE__,                                 \
                "CHECK_THROWS_AS(" #expr ", " #exception_type ") threw a different exception"); \
        }                                                                       \
        if (!thrown_)                                                           \
            odsp_test::fail(__FILE__, __LINE__,                                 \
                "CHECK_THROWS_AS(" #expr ", " #exception_type ") did not throw"); \
    } while (0)

#endif // ODSP_TESTFRAMEWORK_HPP
