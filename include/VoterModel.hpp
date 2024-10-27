#include "BaseGraph.hpp"
#include "RegularLattice.hpp"

//using namespace std;

// L'IDEA POTREBBE ESSERE QUELLA DI SOSTITUIRE IL REGULARLATTICE CON UN ALTRO ELEMENTO TEMOPLATE CHE POI SARà IL DIVERSO GRAFO A SECONDA DELLE EVENIENZE
// PER ESEMPIO SI POTREBBE AVERE UN PUNTATORE AD UNA GENERICA CLASSE bASEgRAPH A POSTO DI POPULATION (COME DOVEVA ESSERE IN PRINCIPIO)

template <typename T>
class VoterModel {

    public:
        VoterModel(const RegularLattice<T>&, std::mt19937*);
        VoterModel(RegularLattice<T>&&, std::mt19937*);
        //VoterModel(); // per ora non metto anche una versione in cui il reticolo viene riempito dal modello, lo faccio fare da funzioni a parte da mettere in delle utils


        RegularLattice<T>& get_lattice() const; // il const potrebbe essere problematico
        double get_magnetization() const;
        double get_magnetization_per_site() const;
        int get_mc_steps() const;
        int get_spin_flips() const;
        bool is_consensus_reached() const;

        void set_rng(std::mt19937*);

        void run_montecarlo(int, int, bool);
        
    private:
        void _compute_magnetization();
        void _single_spin_flip();
        void _mc_step();

        RegularLattice<T> _population;
        double _magnetization;
        int _executed_mc_steps, _executed_spin_flips; // POTREBBERO ESSERE TROPPO PICCOLI (ALMENO IL SECONDO)
        std::mt19937 *_rng;
};