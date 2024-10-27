#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <numeric>

#include "BaseGraph.hpp"
#include "RegularLattice.hpp"
#include "VoterModel.hpp"

using namespace std;

// Constructors
template <typename T>
VoterModel<T>::VoterModel(const RegularLattice<T>& population, std::mt19937* rng) 
    : _population(population),
      _magnetization(0.),
      _executed_mc_steps(0),
      _executed_spin_flips(0),
      _rng(rng)
{}

template <typename T>
VoterModel<T>::VoterModel(RegularLattice<T>&& population, std::mt19937* rng)
    : _population(population),
      _magnetization(0.),
      _executed_mc_steps(0),
      _executed_spin_flips(0),
      _rng(rng)
{}

template <typename T>
RegularLattice<T>& VoterModel<T>::get_lattice() const { return _population.get_graph(); }

template <typename T>
double VoterModel<T>::get_magnetization() const { return _magnetization; };

template <typename T>
double VoterModel<T>::get_magnetization_per_site() const {

    //double mag_per_site = _magnetization / accumulate(_population.get_dimensions().begin(), _population.get_dimensions().end(), std::multiply<int>());
    double mag_per_site = _magnetization / _population.get_linearized_dimension();
    
    return mag_per_site;
}

template <typename T>
int VoterModel<T>::get_mc_steps() const { return _executed_mc_steps; };

template <typename T>
int VoterModel<T>::get_spin_flips() const { return _executed_spin_flips; };

template <typename T>
void VoterModel<T>::set_rng(std::mt19937 *rng) { _rng = rng; };

template <typename T>
void VoterModel<T>::_compute_magnetization() {

    //this->_magnetization = accumulate(_population.get_graph().begin(), _population.get_graph().end(), std::add<T>());
    _magnetization = accumulate(_population.get_graph().begin(), _population.get_graph().end(), std::plus<T>());
}

// QUI POTREBBE ESSERE NECESSARIO FAR CALCOLARE PRIMA LA MAGNETIZZAZIONE
template <typename T>
bool VoterModel<T>:: is_consensus_reached() const {

    bool is_reached = false;

    if (CompareDouble(this->get_magnetization_per_site(), 1.))
        is_reached = true;

    return is_reached;
}

// Esegue il flip di un singolo spin basandosi su un vicino random
// Si può alleggerire notevolmente richiedendo solo la posizione del vettore linearizzato, MA serve una funzione che la riconverta in vettore
template <typename T>
void VoterModel<T>::_single_spin_flip() {

    vector<int> rnd_pos;
    //da vedere se è molto pesante farlo ogni volta, nel caso si aggiunge una variabile alla classe e la si mette nel costruttore

    // Get random spin (position)
    for (int d : _population.get_dimensions()) {
        std::uniform_int_distribution<int> distribution(0, d - 1);

        rnd_pos.push_back(distribution(*_rng));     
    }

    // Get random neighbour
    vector<T> neightbours = _population.get_neighbours(rnd_pos);
    std::uniform_int_distribution<int> distribution(0, neightbours.size());

    int i = distribution(*_rng);

    _population.set_site(rnd_pos, neightbours[i]);

    // Update number of spin flips
    _executed_spin_flips++;
}

// Uno step montacarlo equivale a N step singoli (dove N è la dimensione linearizzata del reticolo)
template <typename T>
void VoterModel<T>::_mc_step() {

    for (unsigned i = 0; i < _population.get_linearized_dimension(); i++)
        _single_spin_flip();
    
    _executed_mc_steps++;
}

// se il bool è falso allora si riferisce al numero di spin flip, altrimenti al numero di step montecarlo
// QUI DENTRO SERVIRà SICURAMENTE METTERE QUALCOSA PER CALCOLARE LA MAGNETIZZAZIONE O GLI ALTRI PARAMETRI PER VEDERE I RISULTATI
// Satmpo ogni 10% un log
template <typename T>
//void VoterModel<T>::run_montecarlo(int steps, int log_percentage=10, bool use_mc_steps=true) {
void VoterModel<T>::run_montecarlo(int steps, int log_percentage, bool use_mc_steps) {

    for (int i = 0; i < steps; i++) {
        if (use_mc_steps) {
            _mc_step();
        }
        else {
            _single_spin_flip();
        }

        // Log every 10 %
        // GUARDA SE AGGIUNGERE ALTRE PARTI AL LOG
        int executition_percentage = static_cast<int> (100*(1. * i / steps));
            
        if (executition_percentage % log_percentage == 0) {
            cout << "Execution percentage: " << executition_percentage << "%" << endl;
        }
        
    }
    

}











// Crea un reticolo con la frazione fornita di spin up e il resto down
vector<int> CreateRandomLattice(unsigned length, float up_fraction, std::mt19937 *rng) {

    if (up_fraction < 0.f or up_fraction > 1.f) {
        cerr << "ERROR: Invalid fraction of positive spins. Required 0 < up_fr < 1, but given up_fr = " << up_fraction << endl;
        exit(1);
    }

    vector<int> lattice(length, 0);
    unsigned n_up = (unsigned) (up_fraction * length);

    // Fill vector
    for (unsigned i = 0; i < length; i++)
        lattice[i] = (i < n_up) ? 1 : -1;

    // Apply random shuffle
    shuffle(lattice.begin(), lattice.end(), *rng);

    return lattice;   
}

// Uso sia per float che per double
template <typename T>
inline bool CompareDouble(T inValue, T inCompareTo) {

    return std::fabs(inValue - inCompareTo) <= std::numeric_limits<double>::epsilon();
}



int main() {
   
    // Generatore
    std::mt19937 rnd_gen; 

    // Reticolo
    vector<int> dim = {2,3,4};
    RegularLattice<int> lattice(CreateRandomLattice(2*3*4, 0.7, &rnd_gen), dim);

    // Modello
    VoterModel<int> model(lattice, &rnd_gen);

    model.run_montecarlo(100, 10, false);

    cout << "Magnetization: " << model.get_magnetization() << " per site: " << model.get_magnetization_per_site() << endl;

    return 0;
}