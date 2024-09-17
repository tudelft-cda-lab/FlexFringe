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
    static double compute_fitness_geo_mean(vector<apta_node*> state_sequence, bool weighted);
    static double compute_fitness_state_size(vector<apta_node*> state_sequence, bool weighted);
    static double compute_fitness_lower_median(vector<apta_node*> state_sequence);
    static double compute_fitness_lower_median_overall(apta_node* root, vector<apta_node*> state_sequence);
    static double compute_fitness_loop(vector<apta_node*> state_sequence);
    static double compute_median(vector<double> state_sizes);
    static vector<double> get_all_state_sizes(apta_node* root);
    static map<int, int> compute_state_visits_sequence(vector<apta_node*> state_sequence);
};

#endif