#ifndef __EA_UTILS_H__
#define __EA_UTILS_H__

#include <cmath>
#include "apta.h"
using namespace std;

class EA_utils {

public:
    EA_utils();
    ~EA_utils();
    static vector<double> compute_fitnesses(vector<vector<apta_node*>> state_sequences, apta_node* root, string type);

private:
    static double compute_fitness_min(vector<apta_node*> state_sequence);
    static double compute_fitness_avg(vector<apta_node*> state_sequence);
    static double compute_fitness_entropy(vector<apta_node*> state_sequence, apta_node* root);
    static double calculate_information_entropy(vector<apta_node*> state_sequence,apta_node* root);
};

#endif