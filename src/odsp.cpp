// odsp: run any model of the library on any topology from the command line.
// See `odsp --help`; all the logic lives in src/driver/.
#include "driver/Driver.hpp"

int main(int argc, char** argv) {
    return driver::run_driver(argc, argv);
}
