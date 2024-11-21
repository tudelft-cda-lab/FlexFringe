/**
 * @file probabilistic_heuristic_interface.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "probabilistic_heuristic_interface.h"

using namespace std;

/**
 * @brief Used for merge tests. Takes two nodes left and right. When merging right into left the access traces of 
 * the right node remains the same, but when parsing the automaton the last (final) probability will change. That 
 * means that if we want to perform the merge tests as described in e.g. "PDFA Distillation via String Probability 
 * Queries", Baumgartner and Verwer 2024, and "PDFA Distillation with Error Bound Guarantees", Baumgartner and Verwer 2024, 
 * then we only have to multiply the access trace's probability path with the final probability of left node. 
 * This function compares those two probabilities and returns the absolute value of the difference.
 */
double probabilistic_heuristic_interface::get_merge_distance_access_trace(apta* aut, apta_node* left_node, apta_node* right_node){
    auto* l = static_cast<probabilistic_heuristic_interface_data*>( left_node->get_data() );
    auto* r = static_cast<probabilistic_heuristic_interface_data*>( right_node->get_data() );

    auto right_sequence = right_node->get_access_trace()->get_input_sequence(true, false);
    auto n = aut->get_root();
    auto n_data = static_cast<probabilistic_heuristic_interface_data*>( n->get_data() );

    double at_weight = 1;
    for(auto symbol: right_sequence){
        at_weight *= n_data->get_probability(symbol);
        n = n->get_child(symbol);
        n_data = static_cast<probabilistic_heuristic_interface_data*>( n->get_data() );
    }

    double diff = abs(at_weight * l->get_final_probability() - at_weight * r->get_final_probability());
    return diff;
};