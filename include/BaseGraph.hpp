#ifndef _BaseGraph_hpp_
#define _BaseGraph_hpp_

#include <vector>

using namespace std;

// Una possibilità è di passare a Boost.MultiArray per il vettore -> potrebbe essere più performante
template <typename T>
class BaseGraph {
    public:
        BaseGraph(){;};
        ~BaseGraph(){;};

        // forse qui serve mettere anche i get e i set
        //virtual  get_element
        //virtual  get_element
        virtual vector<T> get_neighbours(const vector<int>&) const = 0; //per ora lascio il vettore, ma sarebbe meglio farlo con un int 
                                                           //posso anche farne due versioni, una interna/privata con gli int e una pubblica con vector
};

#endif