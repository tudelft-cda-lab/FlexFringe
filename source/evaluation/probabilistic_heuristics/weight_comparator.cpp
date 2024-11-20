/**
 * @file weight_comparator.cpp
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
#include "weight_comparator.h"
#include "parameters.h"
#include "input/inputdatalocator.h"

#include <iostream>
#include <unordered_set>
#include <cmath>

REGISTER_DEF_DATATYPE(weight_comparator_data);
REGISTER_DEF_TYPE(weight_comparator);


void weight_comparator_data::read_json(json& data){
    evaluation_data::read_json(data);

    static int alphabet_size = inputdata_locator::get()->get_alphabet().size();
    symbol_probability_map.resize(alphabet_size);

    json& d = data["trans_probs"];
    for (auto& symbol : d.items()){
        std::string sym = symbol.key();
        std::string val = symbol.value();
        auto tmp = std::stof(val);
        symbol_probability_map[inputdata_locator::get()->symbol_from_string(sym)] = exp(tmp)-1;
    }

    std::string fp = data["final_prob"];
    final_prob = exp(std::stof(fp))-1;
};

void weight_comparator_data::write_json(json& data){
    evaluation_data::write_json(data);

    for(int symbol=0; symbol<symbol_probability_map.size(); ++symbol) {
        auto value = symbol_probability_map[symbol];
        
        std::stringstream ss;
        ss << log(value+1);
        data["trans_probs"][inputdata_locator::get()->string_from_symbol(symbol)] = ss.str();
    }
    
    data["final_prob"] = std::to_string(log(final_prob+1));
};

void weight_comparator_data::print_transition_label(std::iostream& output, int symbol){
    output << symbol << ": " << symbol_probability_map[symbol];
};

void weight_comparator_data::print_state_label(std::iostream& output){
    evaluation_data::print_state_label(output);
    if(DEBUGGING){
        for(int symbol=0; symbol<symbol_probability_map.size(); ++symbol){
            output << symbol << " : " << symbol_probability_map[symbol] << "\n";
        }
    }
    output << "f: " << final_prob << "\n";
};

/** Merging update and undo_merge routines */

void weight_comparator_data::update(evaluation_data* right){
    evaluation_data::update(right);
};

void weight_comparator_data::undo(evaluation_data* right){
    evaluation_data::undo(right);
};

/**
 * @brief Finds the max-symbol in this node.
 * 
 * @return int The symbol with the maximum probability.
 */
int weight_comparator_data::predict_symbol(tail*){
    double max_p = -1;
    int max_symbol = 0;
    for(int s=0; s<symbol_probability_map.size(); ++s){
        auto p = symbol_probability_map[s];
        if(max_p < 0 || max_p < p){
            max_p = p;
            max_symbol = s;
        }
    }
    return max_symbol;
};

double weight_comparator_data::predict_symbol_score(int t){
    if(t == -1)
        return final_prob == 0.0 ? -100.1 : log(final_prob); // if final_prob is zero give very low probability (see log scale)
    return symbol_probability_map[t] == 0.0 ? -100.1 : log(symbol_probability_map[t]);
}

bool weight_comparator::consistent(state_merger *merger, apta_node* left_node, apta_node* right_node, int depth){
    if(inconsistency_found) return false;
    
    static const double mu = MU;

    auto* l = static_cast<weight_comparator_data*>( left_node->get_data() );
    auto* r = static_cast<weight_comparator_data*>( right_node->get_data() );

    auto right_sequence = right_node->get_access_trace()->get_input_sequence(true, false);
    auto n = merger->get_aut()->get_root();
    auto n_data = static_cast<weight_comparator_data*>( n->get_data() );

    double at_weight = 1;
    for(auto symbol: right_sequence){
        at_weight *= n_data->get_probability(symbol);
        n = n->get_child(symbol);
        n_data = static_cast<weight_comparator_data*>( n->get_data() );
    }

    at_weight *= l->get_final_probability();

    double diff = abs(at_weight - r->access_trace_prob);
    if(diff > mu){
        inconsistency_found = true;
        return false;
    }

    score = -diff;
    return true;
};

double weight_comparator::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    return score;
};

void weight_comparator::reset(state_merger *merger){
    inconsistency_found = false;
    score = 0;
};
