CXX=g++
CXXFLAGS=-std=c++11 -Wall -O 
GRAPHOBJ=RegularLattice.o
DEPS=BaseGraph.hpp
GOAL=$(MAKECMDGOALS)
OBJECTS=$(GOAL).o $(RANDOBJ)
	
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

%: $(OBJECTS) #%.o $(OBJ) $(DEPS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(MAKECMDGOALS)

$(RANDOBJ): RegularLattice.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@


.PHONY: clean

clean:
	rm -v *.o