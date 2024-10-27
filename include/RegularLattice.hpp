#ifndef _RegularLattice_hpp_
#define _RegularLattice_hpp_

#include <vector>

#include "BaseGraph.hpp"

using namespace std;

template <typename T>
class RegularLattice : public BaseGraph<T> {
    public:
        RegularLattice(){;};
        RegularLattice(const vector<T>&, const vector<int>&);
        RegularLattice(vector<T>&&, vector<int>&&);
        // !!!!!!!!!!! METTI UN COPY E UN MOVE CONSTRUCTOR
        // fai un costruttore che prende 2 vettori: dimensione, reticolo (con check che siano compatibili) -> magari questo check può essere fatto da una funzione privata
        // fai un costruttore che prende 1 vettori: dimensione
        // fai anche le versioni con le right value reference (almeno per il lattice)

        T get_site(const vector<int>&) const;
        vector<int> get_dimensions() const;
        unsigned get_linearized_dimension() const;
        vector<T> get_neighbours(const vector<int>&) const override; //guarda se qui inserire un bool per scegliere se usare le PBC 
                                                                // potrei anche mettere sia il const nel metodo che nella unzione che ritorna, ma non sarebbe più modificabile
        vector<T>& get_graph() const; //potrebbe dare problemi il const

        void set_site(const vector<int>&, T);

    private:
        T _return_lattice_site(const vector<int>&) const;
        bool _is_dim_compatible(const vector<int>&) const;
        int _pbc(int,int) const;
        //vector<int>& pbc(const vector<int>&) const;
         
        vector<T> _lattice;
        vector<int> _dimensions;
};

#endif