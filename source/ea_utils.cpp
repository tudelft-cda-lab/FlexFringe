#include "ea_utils.h"

double EA_utils::compute_fitness_min(vector<apta_node*> state_sequence) {
    double num_states_visited = 0.0;
    double min_state_size = 100000000.0;
    for (int i = 1; i < state_sequence.size(); i++) {
        num_states_visited++;
        double weighted_state_size = state_sequence[i]->get_size() * (double) i;
        if (weighted_state_size < min_state_size) {
            min_state_size = weighted_state_size;
        }
    }

    return 1.0 / min_state_size;
}

double EA_utils::compute_fitness_avg(vector<apta_node*> state_sequence) {
    double num_states_visited = 0.0;
    double sum_state_sizes = 0.0;
    for (int i = 1; i < state_sequence.size(); i++) {
        num_states_visited++;
        sum_state_sizes += state_sequence[i]->get_size();
    }
    
    if (sum_state_sizes == 0.0 || num_states_visited == 0.0) {
        return 0.0;
    }

    return num_states_visited / sum_state_sizes;

}

double EA_utils::compute_fitness_geo_mean(vector<apta_node*> state_sequence, bool weighted) {
    double prod = 1.0;
    map<int, int> sequence_state_visits;
    if (weighted) {
        sequence_state_visits = EA_utils::compute_state_visits_sequence(state_sequence);
    }

    for (int i = 1; i < state_sequence.size(); i++) {
        double state_visits = state_sequence[i]->get_size();
        if (weighted) {
            prod *= ((1.0/ (double) sequence_state_visits[state_sequence[i]->get_number()]) * (double) (1.0 / state_visits));
        }
        else {
            prod *= (double) (1.0 / state_visits);
        
        }
    }
    
    double ret = (double) pow(prod, (double) 1/ (double) (state_sequence.size() - 1));

    return ret / ((double) state_sequence.size()-1);
}

vector<double> EA_utils::compute_fitnesses(vector<vector<apta_node*>> state_sequences, apta_node* root, string type) {
    vector<double> fitnesses;
    for (auto sequence : state_sequences) {
        if (type == "min") {
            fitnesses.push_back(EA_utils::compute_fitness_min(sequence));
        } else if (type == "avg") {
            fitnesses.push_back(EA_utils::compute_fitness_avg(sequence));
        } else if (type == "geo_mean") {
            fitnesses.push_back(EA_utils::compute_fitness_geo_mean(sequence, false));
        } else if (type == "geo_mean_weighted") {
            fitnesses.push_back(EA_utils::compute_fitness_geo_mean(sequence, true));
        } else if (type == "geo_mean_normalized") {
            fitnesses.push_back(EA_utils::compute_fitness_geo_mean(sequence, false));
        } else if (type == "lower_median") {
            fitnesses.push_back(EA_utils::compute_fitness_lower_median(sequence));
        } else if (type == "lower_median_overall") {
            fitnesses.push_back(EA_utils::compute_fitness_lower_median_overall(root, sequence));
        } else if (type == "loop") {
            fitnesses.push_back(EA_utils::compute_fitness_loop(sequence));
        } else if (type == "weighted_size") {
            fitnesses.push_back(EA_utils::compute_fitness_state_size(sequence, true));
        }
    }

    return fitnesses;
}



double EA_utils::compute_fitness_state_size(vector<apta_node*> state_sequence, bool weighted) {
    double denom = 0.0;
    vector<double> state_sizes;
    for (int i = 1; i < state_sequence.size(); i++) {
        state_sizes.push_back(state_sequence[i]->get_size());
    }

    sort(state_sizes.begin(), state_sizes.end());

    for (int i = 0; i < state_sizes.size(); i++) {
        if (weighted) {
            denom += ((double) (i + 1) * state_sizes[i]);
        } else {
            denom += state_sizes[i];
        }
    }

    return 1.0 / denom;
}



double EA_utils::compute_fitness_lower_median(vector<apta_node*> state_sequence) {
    double num_states_lower_than_median = 0.0;

    if (state_sequence.size() == 2) {
        return 1 / (double) state_sequence[1]->get_size();
    }

    vector<double> state_sizes;
    for (int i = 1; i < state_sequence.size(); i++) {
        state_sizes.push_back(state_sequence[i]->get_size());
    }


    sort(state_sizes.begin(), state_sizes.end());
    double median = compute_median(state_sizes);

    for (int i = 0; i < state_sizes.size(); i++) {
        if (state_sizes[i] < median) {
            num_states_lower_than_median += 1.0;
        }
        else {
            break;
        }
    }
    
    return num_states_lower_than_median / (double) state_sequence.size();
}


double EA_utils::compute_fitness_lower_median_overall(apta_node* root, vector<apta_node*> state_sequence) {
    double num_states_lower_than_median = 0.0;
    vector<double> sequence_state_sizes;
    for (int i = 1; i < state_sequence.size(); i++) {
        sequence_state_sizes.push_back(state_sequence[i]->get_size());
    }

    sort(sequence_state_sizes.begin(), sequence_state_sizes.end());

    double overall_median = compute_median(get_all_state_sizes(root));

    for (int i = 0; i < sequence_state_sizes.size(); i++) {
        if (sequence_state_sizes[i] < overall_median) {
            num_states_lower_than_median += 1.0;
        }
        else {
            break;
        }
    }

    return num_states_lower_than_median / (double) state_sequence.size();
}


double EA_utils::compute_fitness_loop(vector<apta_node*> state_sequence) {
    double num_loops = 0.0;
    set<int> visited_states;
    for (int i = 1; i < state_sequence.size(); i++) {
        if (visited_states.contains(state_sequence[i]->get_number())) {
            num_loops++;
        }
        visited_states.insert(state_sequence[i]->get_number());
    }

    return num_loops / (double) state_sequence.size();
}

map<int, int> EA_utils::compute_state_visits_sequence(vector<apta_node*> state_sequence) {
    map<int, int> state_visits;
    for (int i = 1; i < state_sequence.size(); i++) {
        if (state_visits.find(state_sequence[i]->get_number()) == state_visits.end()) {
            state_visits[state_sequence[i]->get_number()] = 1;
        } else {
            state_visits[state_sequence[i]->get_number()]++;
        }
    }
    return state_visits;
}

vector<double> EA_utils::get_all_state_sizes(apta_node* root) {
    vector<double> state_sizes;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
          apta_node *n = *Ait;
          if (n->get_number() == -1) {
                continue; // skip the root
          }
          state_sizes.push_back(n->get_size());
    }

    return state_sizes;
}

double EA_utils::compute_median(vector<double> state_sizes) {
    sort(state_sizes.begin(), state_sizes.end());
    if (state_sizes.size() % 2 == 0) {
        return (state_sizes[(int) (state_sizes.size() / 2)] + state_sizes[(int) (state_sizes.size()/ 2) - 1]) / 2.0;
    } else {
        return state_sizes[(int) (state_sizes.size() / 2)];
    }
}

