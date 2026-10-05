CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pthread -Iinclude

# BUILD=release (default): optimised binaries in bin/
# BUILD=debug: -O0 -g with AddressSanitizer + UBSan, binaries in bin/debug/
BUILD ?= release
ifeq ($(BUILD),debug)
  CXXFLAGS += -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer
  BINDIR   := bin/debug
else ifeq ($(BUILD),release)
  CXXFLAGS += -O2
  BINDIR   := bin
else
  $(error BUILD must be 'release' or 'debug', got '$(BUILD)')
endif

# The git commit is baked into the binaries (see include/Version.hpp)
GIT_VERSION   := $(shell git describe --always --dirty 2>/dev/null || echo unknown)
VERSION_STAMP := $(BINDIR)/.git_version
$(shell mkdir -p $(BINDIR) && (echo '$(GIT_VERSION)' | cmp -s - $(VERSION_STAMP) || echo '$(GIT_VERSION)' > $(VERSION_STAMP)))
CXXFLAGS += -DODSP_GIT_VERSION='"$(GIT_VERSION)"'

SRCDIR   := src
TESTDIR  := tests
HEADERS  := $(wildcard include/*.hpp) $(VERSION_STAMP)
DRIVER   := $(wildcard $(SRCDIR)/driver/*.hpp)
TEST_SRC := $(wildcard $(TESTDIR)/*.cpp)
TEST_HDR := $(wildcard $(TESTDIR)/*.hpp)

all: $(BINDIR)/odsp $(BINDIR)/basics $(BINDIR)/ensemble $(BINDIR)/noisy $(BINDIR)/networks $(BINDIR)/models $(BINDIR)/opinions $(BINDIR)/continuous

$(BINDIR):
	mkdir -p $@

# The command-line driver (src/odsp.cpp + src/driver/).
$(BINDIR)/odsp: $(SRCDIR)/odsp.cpp $(DRIVER) $(HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) $< -o $@

$(BINDIR)/basics: $(SRCDIR)/basics_demo.cpp $(HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BINDIR)/ensemble: $(SRCDIR)/ensemble_demo.cpp $(HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BINDIR)/noisy: $(SRCDIR)/noisy_demo.cpp $(HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BINDIR)/networks: $(SRCDIR)/networks_demo.cpp $(HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BINDIR)/models: $(SRCDIR)/models_demo.cpp $(HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BINDIR)/opinions: $(SRCDIR)/opinions_demo.cpp $(HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BINDIR)/continuous: $(SRCDIR)/continuous_demo.cpp $(HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# All test files build into one binary; run a subset with
#   make test FILTER=<substring of test-case names>
TEST_OBJ := $(patsubst $(TESTDIR)/%.cpp,$(BINDIR)/obj/tests/%.o,$(TEST_SRC))

$(BINDIR)/obj/tests:
	mkdir -p $@

$(BINDIR)/obj/tests/%.o: $(TESTDIR)/%.cpp $(TEST_HDR) $(HEADERS) $(DRIVER) | $(BINDIR)/obj/tests
	$(CXX) $(CXXFLAGS) -I$(TESTDIR) -I$(SRCDIR) -c $< -o $@

$(BINDIR)/tests: $(TEST_OBJ) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(TEST_OBJ) -o $@

.PHONY: all test debug run run-basics run-ensemble run-noisy run-networks run-models run-opinions run-continuous clean
test: $(BINDIR)/tests
	./$< $(FILTER)

# Build everything with sanitizers and run the tests under them
debug:
	$(MAKE) BUILD=debug all test

run: $(BINDIR)/odsp
	./$< $(ARGS)
run-basics: $(BINDIR)/basics
	./$< $(ARGS)
run-ensemble: $(BINDIR)/ensemble
	./$< $(ARGS)
run-noisy: $(BINDIR)/noisy
	./$< $(ARGS)
run-networks: $(BINDIR)/networks
	./$< $(ARGS)
run-models: $(BINDIR)/models
	./$< $(ARGS)
run-opinions: $(BINDIR)/opinions
	./$< $(ARGS)
run-continuous: $(BINDIR)/continuous
	./$< $(ARGS)

clean:
	rm -rf bin
