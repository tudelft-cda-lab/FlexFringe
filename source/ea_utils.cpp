#include "ea_utils.h"

double EA_utils::compute_fitness_min(vector<apta_node*> state_sequence) {
    double num_states_visited = 0.0;
    double min_state_size = 100000000.0;
    for (int i = 1; i < state_sequence.size(); i++) {
        num_states_visited++;
        int state_size = state_sequence[i]->get_size();
        if (state_size < min_state_size) {
            min_state_size = state_size;
        }
    }

    return num_states_visited / min_state_size;
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

double EA_utils::compute_fitness_entropy(vector<apta_node*> state_sequence, apta_node* root) {
    double entropy = calculate_information_entropy(state_sequence, root);
    if (entropy == 0.0) {
        return 0.0;
    }
    return entropy;
}

double EA_utils::calculate_information_entropy(vector<apta_node*> state_sequence, apta_node* root) {
    double sum_state_visits = 0.0;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
          apta_node *n = *Ait;
          if (n->get_number() == -1) {
                continue; // skip the root
          }
          sum_state_visits += n->get_size();
        }
    
    double entropy = 0.0;
    double sum_prob = 0.0;
    set<int> visited_states;
    for (int i = 1; i < state_sequence.size(); i++) {
        if (visited_states.contains(state_sequence[i]->get_number())) {
            continue;
        }
        double state_visits = state_sequence[i]->get_size();
        double state_prob = state_visits / sum_state_visits;
        sum_prob += state_prob;
        entropy += (state_prob * log2(state_prob));
        visited_states.insert(state_sequence[i]->get_number());
    }

    entropy += ((1 - sum_prob) * log2(1 - sum_prob));
    return -entropy;
}

double EA_utils::compute_fitness_geo_mean(vector<apta_node*> state_sequence) {
    double prod = 1.0;
    for (int i = 1; i < state_sequence.size(); i++) {
        double state_visits = state_sequence[i]->get_size();
        prod *= (double) (1 / state_visits);
    }
    return (double) pow(prod, (double) 1/ (double) (state_sequence.size() - 1));
}

vector<double> EA_utils::compute_fitnesses(vector<vector<apta_node*>> state_sequences, apta_node* root, string type) {
    vector<double> fitnesses;
    for (auto sequence : state_sequences) {
        if (type == "min") {
            fitnesses.push_back(EA_utils::compute_fitness_min(sequence));
        } else if (type == "avg") {
            fitnesses.push_back(EA_utils::compute_fitness_avg(sequence));
        } else if (type == "entropy") {
            fitnesses.push_back(EA_utils::compute_fitness_entropy(sequence, root));
        }
        else if (type == "geo_mean") {
            fitnesses.push_back(EA_utils::compute_fitness_geo_mean(sequence));
        }
    }

    return fitnesses;
}


