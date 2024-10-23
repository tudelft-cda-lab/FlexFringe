#ifndef __EA_UTILS_H__
#define __EA_UTILS_H__

#include <cmath>
#include "apta.h"
#include <tuple>
using namespace std;

class EA_utils {

public:
    EA_utils();
    ~EA_utils();
    static vector<double> compute_fitness_values(vector<vector<tuple<int, int>>> state_sequences, apta_node* root, string type);

private:
    static double compute_fitness_min(vector<tuple<int, int>> state_sequence);
    static double compute_fitness_avg(vector<tuple<int, int>> state_sequence);
    static double compute_fitness_geo_mean(vector<tuple<int, int>> state_sequence, bool weighted);
    static double compute_fitness_weighted_size(vector<tuple<int, int>> state_sequence, bool weighted);
    static double compute_fitness_lower_median(vector<tuple<int, int>> state_sequence);
    static double compute_fitness_lower_median_overall(vector<vector<tuple<int, int>>> state_sequences, vector<tuple<int, int>> state_sequence);
    static double compute_fitness_loop(vector<tuple<int, int>> state_sequence);
    static double compute_median(vector<double> state_sizes);
    static vector<double> get_all_state_sizes(vector<vector<tuple<int, int>>> state_sequences);
    static map<int, int> compute_state_visits_sequence(vector<tuple<int, int>> state_sequence);
};

#endif