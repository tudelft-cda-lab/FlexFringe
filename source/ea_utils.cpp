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

double EA_utils::compute_fitness_entropy(vector<apta_node*> state_sequence, apta_node* root, bool weighted) {
    double entropy = calculate_information_entropy(state_sequence, root, weighted);
    if (entropy == 0.0) {
        return 0.0;
    }
    return entropy;
}

double EA_utils::calculate_information_entropy(vector<apta_node*> state_sequence, apta_node* root, bool weighted) {
    double sum_state_visits = EA_utils::compute_sum_visits(root);
    double entropy = 0.0;
    double sum_prob = 0.0;
    map<int, int> sequence_state_visits;
    
    if (weighted) {
        sequence_state_visits = EA_utils::compute_state_visits_sequence(state_sequence);
    }

    set<int> visited_states;
    for (int i = 1; i < state_sequence.size(); i++) {
        if (visited_states.contains(state_sequence[i]->get_number())) {
            continue;
        }

        double state_visits = state_sequence[i]->get_size();
        double state_prob = state_visits / sum_state_visits;
        sum_prob += state_prob;

        if (weighted) {
            entropy += ((1.0 / (double) sequence_state_visits[state_sequence[i]->get_number()]) * state_prob * log2(state_prob));
        }
        else {
            entropy += (state_prob * log2(state_prob));
        }

        visited_states.insert(state_sequence[i]->get_number());
    }

    entropy += ((1 - sum_prob) * log2(1 - sum_prob));
    return -entropy;
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
        } else if (type == "entropy") {
            fitnesses.push_back(EA_utils::compute_fitness_entropy(sequence, root, false));
        } else if (type == "geo_mean") {
            fitnesses.push_back(EA_utils::compute_fitness_geo_mean(sequence, false));
        } else if (type == "perplexity") {
            fitnesses.push_back(EA_utils::compute_fitness_perplexity(sequence, root, false));
        } else if (type == "entropy_weighted") {
            fitnesses.push_back(EA_utils::compute_fitness_entropy(sequence, root, true));
        } else if (type == "perplexity_weighted") {
            fitnesses.push_back(EA_utils::compute_fitness_perplexity(sequence, root, true));
        } else if (type == "geo_mean_weighted") {
            fitnesses.push_back(EA_utils::compute_fitness_geo_mean(sequence, true));
        } else if (type == "geo_mean_normalized") {
            fitnesses.push_back(EA_utils::compute_fitness_geo_mean(sequence, false));
        }
    }

    return fitnesses;
}

double EA_utils::compute_fitness_perplexity(vector<apta_node*> state_sequence, apta_node* root, bool weighted) {
    double sum_state_visits = EA_utils::compute_sum_visits(root);
    double perplexity = 1.0;
    map<int, int> sequence_state_visits;
    if (weighted) {
        sequence_state_visits = EA_utils::compute_state_visits_sequence(state_sequence);
    }

    for (int i = 1; i < state_sequence.size(); i++) {
        double state_visits = state_sequence[i]->get_size();
        double state_prob = state_visits / sum_state_visits;
        if (weighted) {
            perplexity *= ((1.0/ (double) sequence_state_visits[state_sequence[i]->get_number()]) * (double) (pow(state_prob, -state_prob)));
        } 
        else{
            perplexity *= (double) (pow(state_prob, -state_prob));
        }
    }
    
    return perplexity;
}

double EA_utils::compute_sum_visits(apta_node* root) {
    double sum_visits = 0.0;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
          apta_node *n = *Ait;
          if (n->get_number() == -1) {
                continue; // skip the root
          }
          sum_visits += (double) n->get_size();
    }
    return sum_visits;
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

