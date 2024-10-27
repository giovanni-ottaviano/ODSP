#include <numeric>
#include <vector>
#include <iostream>

#include "RegularLattice.hpp"

using namespace std;

// Constructors
template <typename T>
RegularLattice<T>::RegularLattice(const vector<T>& lattice, const vector<int>& dimensions) 
    : _lattice(lattice),
      _dimensions(dimensions)
{
    int expected_elements = accumulate(dimensions.begin(), dimensions.end(), std::multiplies<T>());
    assert(lattice.size() == expected_elements);                                             //potrebbe essere problemativa la differenza di tipo!!!
}

template <typename T>
RegularLattice<T>::RegularLattice(vector<T>&& lattice, vector<int>&& dimensions) 
    : _lattice(lattice),
      _dimensions(dimensions)
{
    int expected_elements = accumulate(dimensions.begin(), dimensions.end(), std::multiplies<T>());
    assert(lattice.size() == expected_elements);                                             //potrebbe essere problemativa la differenza di tipo!!!
}

// Public methods
template <typename T>
T RegularLattice<T>::get_site(const vector<int>& position) const {
     
     if(!this->_is_dim_compatible(position)) {
        std::cerr << "Lattice is " << this->_dimensions.size() << "-dimensional, but " << position.size() << " dimensions are given." << endl;

        exit(1);
     }

    return this->_return_lattice_site(position);
}

template <typename T>
vector<int> RegularLattice<T>::get_dimensions() const {

    return _dimensions;
}

template <typename T>
unsigned RegularLattice<T>::get_linearized_dimension() const { return _lattice.size();}

template <typename T>
vector<T> RegularLattice<T>::get_neighbours(const vector<int>& position) const {

    vector<T> neighbours(2*position.size());
    int shift[2] = {-1, 1};

    for (unsigned int i = 0; i < position.size(); i++) {
        for (int s : shift) {
            vector<int> copy_position(position);

            // Apply Periodic Boundary Conditions
            copy_position[i] = this->_pbc(copy_position[i] + s, this->_dimensions[i]);

            neighbours.push_back(this->_return_lattice_site(copy_position));
        }
        
    }
    
    return neighbours;
}

template <typename T>
vector<T>& RegularLattice<T>::get_graph() const {

    return _lattice;
}

template <typename T>
void RegularLattice<T>::set_site(const vector<int>& position, T value) {

    this->_return_lattice_site(position) = value;
}

// Private methods
// Return the value of selected site.  The site is passed as a vector (x,y,z,...) and internally linearized to get the correct site
//Forse qui dovrebbe restituire una referenza
template <typename T>
T RegularLattice<T>::_return_lattice_site(const vector<int>& position) const {

    // Lo uso per fare debugging, poi forse conviene toglierlo
    if(!this->_is_dim_compatible(position)) {
        std::cerr << "Lattice is " << this->_dimensions.size() << "-dimensional, but " << position.size() << " dimensions are given." << endl;

        exit(1);
    }
    
    int linear_index = 0;

/*
// Versione C++ - style    
    int i = 1;

    for (auto ix = position.cbegin(); ix != position.cend() - 1; ix++) {
        int accumulate_prod = accumulate(this->_dimensions.cbegin() + i, this->_dimensions.cend(), std::multiplies<T>());

        linear_index += (*ix) * accumulate_prod;
        i++;
    }

    linear_index += position.back();
*/    

//Versione C-style
// POTREBBERO ESSERE PROBLEMATICHE LE LIMITAZIONI NUMERICHE SE SI USANO TANTE DIMENSIONI
    for (unsigned int ix = 0; ix < position.size(); ix++) {
        int accumulate_prod = 1;

        for (unsigned int j = ix + 1; j < position.size(); j++) //stessa dimensione di _dimensions
            accumulate_prod *= this->_dimensions[j];
         
        linear_index += position[ix] * accumulate_prod;
    }

    return this->_lattice.at(linear_index);
}

template <typename T>
bool RegularLattice<T>::_is_dim_compatible(const vector<int>& dimensions) const {

    bool is_compatible = true;

    if (dimensions.size() != this->_dimensions.size()) {
        is_compatible = false;
    }
    else {
        for (unsigned int i = 0; i < dimensions.size(); i++) {
            if (dimensions[i] != this->_dimensions[i]) {
                is_compatible = false;
                break;
            }
        }   
    }
    
    return is_compatible;
}

template <typename T>
int RegularLattice<T>::_pbc(int pos, int dim) const {

    return pos % dim;
}



// int main() {
   
//     RegularLattice<int>* lattice = new RegularLattice<int>();
//     vector<int> a0 = {2,2,3}, a = {2,2,3}, b = {1,2,3}, c = {2,2,3,4};

//     cout << "PBC: " << lattice->_pbc(0,3) << " " << lattice->_pbc(1,3) << " " << lattice->_pbc(2,3) << " " << lattice->_pbc(3 ,3) << " " << lattice->_pbc(4,3) << endl;

//     lattice->_dimensions = a0;

//     cout << "Compatible: " << lattice->_is_dim_compatible(a) << " " << lattice->_is_dim_compatible(b) << " " << lattice->_is_dim_compatible(c) << endl;

//     delete lattice;

//     return 0;
// }