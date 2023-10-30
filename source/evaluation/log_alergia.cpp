/**
 * @file log_alergia.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#include "state_merger.h"
#include "evaluate.h"
#include "log_alergia.h"
#include "parameters.h"
#include "input/inputdatalocator.h"

#include <iostream>
#include <unordered_set>
#include <cmath>

REGISTER_DEF_DATATYPE(log_alergia_data);
REGISTER_DEF_TYPE(log_alergia);


/* void log_alergia_data::initialize() {
    evaluation_data::initialize();
    //symbol_pro.clear();
} */

/* void log_alergia_data::add_tail(tail* t){
} */

/* void log_alergia_data::del_tail(tail* t){
    evaluation_data::del_tail(t);
} */

void log_alergia_data::read_json(json& data){
    evaluation_data::read_json(data);

    static int alphabet_size = inputdata_locator::get()->get_alphabet().size();
    normalized_symbol_probability_map.resize(alphabet_size);

    json& d = data["trans_probs"];
    for (auto& symbol : d.items()){
        string sym = symbol.key();
        string val = symbol.value();
        normalized_symbol_probability_map[inputdata_locator::get()->symbol_from_string(sym)] = stod(val);
    }

    if(FINAL_PROBABILITIES){
        string fp = data["final_prob"];
        final_prob = stod(fp);
    }

    //log_alergia::normalize_probabilities(this);
};

void log_alergia_data::write_json(json& data){
    evaluation_data::write_json(data);


    //for(auto & symbol_count : normalized_symbol_probability_map) {
    for(int symbol=0; symbol<normalized_symbol_probability_map.size(); ++symbol) {
        double value = normalized_symbol_probability_map[symbol];
        data["trans_probs"][inputdata_locator::get()->string_from_symbol(symbol)] = to_string(value);
    }
    
    if(FINAL_PROBABILITIES) data["final_prob"] = to_string(final_prob);
};

void log_alergia_data::print_transition_label(iostream& output, int symbol){
    output << symbol << ": " << symbol_probability_map[symbol];
};

void log_alergia_data::print_state_label(iostream& output){
    evaluation_data::print_state_label(output);
    for(int symbol=0; symbol<normalized_symbol_probability_map.size(); ++symbol){
        output << symbol << " : " << normalized_symbol_probability_map[symbol] << "\n";
    }
    if(FINAL_PROBABILITIES) output << "f: " << final_prob << "\n";
};

/** Merging update and undo_merge routines */

void log_alergia_data::update(evaluation_data* right){
    evaluation_data::update(right);
};

void log_alergia_data::undo(evaluation_data* right){
    evaluation_data::undo(right);
};

/**
 * @brief Finds the max-symbol in this node.
 * 
 * @return int The symbol with the maximum probability.
 */
int log_alergia_data::predict_symbol(tail*){
    double max_p = -1;
    int max_symbol = 0;
    for(int s=0; s<normalized_symbol_probability_map.size(); ++s){
        auto p = normalized_symbol_probability_map[s];
        if(max_p < 0 || max_p < p){
            max_p = p;
            max_symbol = s;
        }
    }
    return max_symbol;
};

double log_alergia_data::predict_symbol_score(int t){
    if(t == -1)
        return final_prob;
    //else if(normalized_symbol_probability_map.contains(t))
    return normalized_symbol_probability_map[t];
    
    //return 0.0;
}

void log_alergia_data::add_probability(const int symbol, const double p) {
    this->symbol_probability_map[symbol] += p;
}

void log_alergia_data::update_probability(const int symbol, const double p) {
    this->symbol_probability_map[symbol] = p;
}

bool log_alergia::consistent(state_merger *merger, apta_node* left_node, apta_node* right_node, int depth){
    if(inconsistency_found) return false;
    
    static const auto mu = static_cast<double>(MU);

    auto* l = static_cast<log_alergia_data*>( left_node->get_data() );
    auto* r = static_cast<log_alergia_data*>( right_node->get_data() );

    if(abs(l->access_trace_prob - r->access_trace_prob) > mu){
      inconsistency_found = true;
      return false;
    }

    auto right_sequence = right_node->get_access_trace()->get_input_sequence(true, false);
    auto n = merger->get_aut()->get_root();
    auto n_data = static_cast<log_alergia_data*>( n->get_data() );

    double at_prob = 1;
    for(auto symbol: right_sequence){
        at_prob *= n_data->get_normalized_probability(symbol);
        n = n->get_child(symbol);
        n_data = static_cast<log_alergia_data*>( n->get_data() );
    }
    at_prob *= l->get_final_prob();

    double diff = abs(at_prob - r->access_trace_prob);
    if(diff > mu){
        inconsistency_found = true;
        return false;
    }

    score = -diff;
    return true;
};

/**
 * @brief Normalizes the symbol probability map, and stores it.
 * 
 * @param node The node.
 */
void log_alergia::normalize_probabilities(log_alergia_data* data) {
    double p_sum = 0;
    for(auto& p: data->symbol_probability_map){
        p_sum += p;
    }

    assert(p_sum != 0);

    double factor = FINAL_PROBABILITIES ? (1. - data->final_prob) / p_sum : 1 / p_sum;
    assert(factor >= 0);

    for(int i=0; i<data->symbol_probability_map.size(); ++i){
        auto p = data->symbol_probability_map[i];
        data->normalized_symbol_probability_map[i] = p * factor;
    }
}

double log_alergia::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    return score;
};

void log_alergia::reset(state_merger *merger){
    inconsistency_found = false;
    score = 0;
};