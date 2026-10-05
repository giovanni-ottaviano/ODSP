#ifndef ODSP_VERSION_HPP
#define ODSP_VERSION_HPP

#include <ctime>
#include <string>

// Version information, recorded in every output file (see CsvWriter's
// preamble) so results can be traced to the code that produced them.
//   ODSP_VERSION      the library version;
//   ODSP_GIT_VERSION  `git describe --always --dirty` at build time, set by the
//                     makefile; "-dirty" means uncommitted changes were built
//                     in. "unknown" when built without the makefile.
#define ODSP_VERSION "0.1.0"
#ifndef ODSP_GIT_VERSION
#define ODSP_GIT_VERSION "unknown"
#endif

inline std::string odsp_version_string() {
    return std::string("ODSP ") + ODSP_VERSION + ", git " + ODSP_GIT_VERSION;
}

// Current local time as ISO 8601 with the UTC offset, e.g. 2026-10-04T19:22:05+02:00.
inline std::string iso_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    char buf[40];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &local);
    std::string s = buf;   // ...+0200 -> ...+02:00
    if (s.size() >= 5) s.insert(s.size() - 2, ":");
    return s;
}

#endif // ODSP_VERSION_HPP
