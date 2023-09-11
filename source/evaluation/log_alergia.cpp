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

    json& d = data["trans_probs"];
    for (auto& symbol : d.items()){
        string sym = symbol.key();
        string val = symbol.value();
        symbol_probability_map[inputdata_locator::get()->symbol_from_string(sym)] = stod(val);
    }
};

void log_alergia_data::write_json(json& data){
    evaluation_data::write_json(data);

    data["symbol_probs"] = {};

    for(auto & symbol_count : symbol_probability_map) {
        int symbol = symbol_count.first;
        double value = symbol_count.second;
        data["trans_probs"][inputdata_locator::get()->string_from_symbol(symbol)] = to_string(value);
    }
};

void log_alergia_data::print_transition_label(iostream& output, int symbol){
    output << symbol_probability_map[symbol] << ": ";
};

/* void log_alergia_data::print_state_label(iostream& output){
    evaluation_data::print_state_label(output);
    //output << "\n" << num_paths() << " " << num_final();
}; */

/** Merging update and undo_merge routines */

void log_alergia_data::update(evaluation_data* right){
    evaluation_data::update(right);
    auto* other = (log_alergia_data*)right;

    for(auto & symbol_p_mapping : other->symbol_probability_map){
        auto& p1 = symbol_probability_map[symbol_p_mapping.first];
        auto& p2 = symbol_p_mapping.second;
        symbol_probability_map[symbol_p_mapping.first] = (p1 + p2) / static_cast<double>(2);
    }
};

void log_alergia_data::undo(evaluation_data* right){
    evaluation_data::undo(right);
    auto* other = (log_alergia_data*)right;

    for(auto & symbol_p_mapping : other->symbol_probability_map){
        auto& p1 = symbol_probability_map[symbol_p_mapping.first];
        auto& p2 = symbol_p_mapping.second;

        symbol_probability_map[symbol_p_mapping.first] = p1 * static_cast<double>(2) - p2; // reverse update-operation
    }
};

/**
 * @brief Finds the max-symbol in this node.
 * 
 * @return int The symbol with the maximum probability.
 */
int log_alergia_data::predict_symbol(tail*){
    double max_p = -1;
    int max_symbol = 0;
    for(auto & symbol_count : symbol_probability_map){
        int count = symbol_count.second;
        if(max_p < 0 || max_p < count){
            max_p = count;
            max_symbol = symbol_count.first;
        }
    }
    return max_symbol;
};

double log_alergia_data::predict_symbol_score(int t){
    if(t == -1)
        return 1.0;
    else if(symbol_probability_map.contains(t))
        return symbol_probability_map[t];
    
    return 0.0;
}

double log_alergia_data::get_probability(const int symbol) {
    return this->symbol_probability_map[symbol];
}

void log_alergia_data::insert_probability(const int symbol, const double p) {
    this->symbol_probability_map[symbol] = p;
}


bool log_alergia::consistent(state_merger *merger, apta_node* left_node, apta_node* right_node){
    if(inconsistency_found) return false;
    
    auto* l = static_cast<log_alergia_data*>( left_node->get_data() );
    auto* r = static_cast<log_alergia_data*>( right_node->get_data() );

    unordered_set<int> checked_symbols;
    for(auto& [symbol, left_p] : l->symbol_probability_map){
        double right_p = r->symbol_probability_map[symbol]; // automatically set to 0 if does not contain (zero initialization)
        double left = log(left_p - right_p);
        if(left > right){
            inconsistency_found = true;
            return false;
        }

        sum_diffs += left;
    }

    return true;
};

double log_alergia::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    return sum_diffs;
};

void log_alergia::reset(state_merger *merger){
    inconsistency_found = false;
    sum_diffs = 0.0;
};

void log_alergia::initialize_before_adding_traces() {
    alpha = CHECK_PARAMETER;
    log_mu = log(MU);
    log_sqr_n = 0.5 * (log(log(2 / alpha)) - log(2)) - log_mu - log(2);

    auto log_two = log(2);
    auto log_log_alpha = log(log(2 / alpha));
    right = 0.5 * (log_log_alpha - log_two) - log_sqr_n - log_two;

    cout << "Evaluation function initialized. Alpha: " << alpha << ", log-mu" << log_mu << ", log(sqrt(n)) [also called Gamma]: " << log_sqr_n << endl;
}