/**
 * @file weight_state_comparator.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-01-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "state_merger.h"
#include "evaluate.h"
#include "weight_state_comparator.h"
#include "parameters.h"
#include "input/inputdatalocator.h"

#include <iostream>
#include <unordered_set>
#include <cmath>

REGISTER_DEF_DATATYPE(weight_state_comparator_data);
REGISTER_DEF_TYPE(weight_state_comparator);

/** Merging update and undo_merge routines */

/* void weight_state_comparator_data::update(evaluation_data* right){
    evaluation_data::update(right);
}; */

/* void weight_state_comparator_data::undo(evaluation_data* right){
    evaluation_data::undo(right);
}; */

double weight_state_comparator::compute_state_distance(apta_node* left_node, apta_node* right_node){
    const auto& left_state = static_cast<weight_state_comparator_data*>( left_node->get_data() )->state;
    const auto& right_state = static_cast<weight_state_comparator_data*>( right_node->get_data() )->state;

    double denominator = 0;
    double left_square_sum = 0, right_square_sum = 0;
    for(int i=0; i<left_state.size(); i++){
        denominator += left_state[i] * right_state[i];

        left_square_sum += left_state[i] * left_state[i];
        right_square_sum += right_state[i] * right_state[i];
    }

    return denominator / ( sqrt(left_square_sum) * sqrt(right_square_sum) );
}

/**
 * @brief What you think it does. Additionally to the weight comparator we also store a state and compare with that one.
 */
bool weight_state_comparator::consistent(state_merger *merger, apta_node* left_node, apta_node* right_node, int depth){
    if(inconsistency_found) return false;
    
    static const auto mu = static_cast<float>(MU);
    static const auto max_state_dist = static_cast<float>(CHECK_PARAMETER);

    if(compute_state_distance(left_node, right_node) > max_state_dist){
        inconsistency_found = true;
        return false;
    } 

    auto* l = static_cast<weight_state_comparator_data*>( left_node->get_data() );
    auto* r = static_cast<weight_state_comparator_data*>( right_node->get_data() );

    auto right_sequence = right_node->get_access_trace()->get_input_sequence(true, false);
    auto n = merger->get_aut()->get_root();
    auto n_data = static_cast<weight_state_comparator_data*>( n->get_data() );

    double at_weight = 1;
    for(auto symbol: right_sequence){
        at_weight *= n_data->get_weight(symbol);
        n = n->get_child(symbol);
        n_data = static_cast<weight_state_comparator_data*>( n->get_data() );
    }

    at_weight *= l->get_final_weight();

    double diff = abs(at_weight - r->access_weight);
    if(diff > mu){
        inconsistency_found = true;
        return false;
    }

    score = -diff;
    return true;
};

double weight_state_comparator::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    return score;
};

void weight_state_comparator::reset(state_merger *merger){
    inconsistency_found = false;
    score = 0;
};

double weight_state_comparator::get_distance(apta* aut, apta_node* left_node, apta_node* right_node){
    auto* l = static_cast<weight_state_comparator_data*>( left_node->get_data() );
    auto* r = static_cast<weight_state_comparator_data*>( right_node->get_data() );

    auto right_sequence = right_node->get_access_trace()->get_input_sequence(true, false);
    auto n = aut->get_root();
    auto n_data = static_cast<weight_state_comparator_data*>( n->get_data() );

    float at_weight = 1;
    for(auto symbol: right_sequence){
        at_weight *= n_data->get_weight(symbol);
        n = n->get_child(symbol);
        n_data = static_cast<weight_state_comparator_data*>( n->get_data() );
    }

    double diff = abs(at_weight * l->get_final_weight() - at_weight * r->get_final_weight());
    return diff;
};