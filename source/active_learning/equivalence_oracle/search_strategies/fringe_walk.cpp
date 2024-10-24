/**
 * @file fringe_walk.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-05-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "fringe_walk.h"

#include "weight_comparator.h"

#include <cassert>
#include <iostream>
#include <list>
#include <vector>

using namespace std;


/**
 * @brief Gets the nodes at the fringe of the underlying unmerged apta.
 * 
 * @return unordered_set<apta_node*> Fringe nodes as set.
 */
unordered_set<apta_node*> fringe_walk::collect_fringe_nodes(){
    const static auto mu = MU;
    const static auto max_length = MAX_CEX_LENGTH;

    unordered_set<apta_node*> res;
    //cout << "apta_iterator" << endl;
    for (APTA_iterator a_it = APTA_iterator(hypothesis->get_root()); *a_it != nullptr; ++a_it){
        auto node = *a_it;
        if(tested_nodes.contains(node)) continue;

        auto data = static_cast<weight_comparator_data*>(node->get_data()); // TODO: make this one a bit more generic or do a check
        if(!node->has_child_nodes() && data->get_access_probability() >= mu && node->get_depth() < max_length /*  && !tested_nodes.contains(node) */)
            res.insert(node);
    }

    return res;
}

/**
 * @brief Gets the next node from the fringe nodes that is not in tested nodes. If cannot be found return nullptr, meaning all nodes have been tested.
 * 
 * We give it the two class attributes in hope the function will execute a bit faster.
 * 
 * @return apta_node* The next node.
 */
apta_node* fringe_walk::get_next_valid_node(const unordered_set<apta_node*>& fringe_nodes, const unordered_set<apta_node*>& tested_nodes) const {
    for(auto n: fringe_nodes){
        if(!tested_nodes.contains(n))
            return n;
    }
    return nullptr;
}

/**
 * @brief What you think it does.
 */
optional<vector<int>> fringe_walk::next(const inputdata& id) {
    static const auto samples_per_node = SAMPLES_PER_NODE;
    const static auto max_length = MAX_CEX_LENGTH;

    if(current_node == nullptr){ // first node after last reset
        current_node = get_next_valid_node(fringe_nodes, tested_nodes);
        if(current_node == nullptr) return nullopt;

        auto at = current_node->get_access_trace();
        current_access_sequence = at->get_input_sequence(true, false);
        
        cout << "Starting with access trace: ";
        for(int x: current_access_sequence) 
            cout << x << " ";
        cout << endl;
    }
        
    if(samples_for_current_node == samples_per_node){
        tested_nodes.insert(current_node);

        current_node = get_next_valid_node(fringe_nodes, tested_nodes);
        if(current_node == nullptr) return nullopt;

        auto at = current_node->get_access_trace();
        current_access_sequence = at->get_input_sequence(true, false);
        
        cout << "Shifting to access trace: ";
        for(int x: current_access_sequence) 
            cout << x << " ";
        cout << endl;

        samples_for_current_node = 0;
        ++n_tested_nodes;
    }

    if(samples_for_current_node % min(samples_per_node, 1000) == 0)
        cout << "Processed nodes:" << n_tested_nodes << "/" << current_h_size << ". Samples: " << samples_for_current_node << endl;

    const int output_string_length = min(length_generator.get_random_int(), max_length-current_node->get_depth());
    vector<int> res(current_access_sequence);
    for (int i = 0; i < output_string_length; ++i) 
        res.push_back(alphabet_vec[alphabet_sampler.get_random_int()]);

    ++samples_for_current_node;
    return res;
}

void fringe_walk::initialize(state_merger* merger){
    this->hypothesis = merger->get_aut();
    this->alphabet_vec = merger->get_dat()->get_alphabet();

    current_node = nullptr;
    
    alphabet_sampler.set_limits(0, alphabet_vec.size() - 1);
    n_tested_nodes = 0;
}

void fringe_walk::reset(){
    if(hypothesis==nullptr){
      cerr << "ERROR: Tried to call reset() method in fringe_walk without initializing the hypothesis (==nullptr). Aborting program." << std::endl;
      throw exception();
    }

    fringe_nodes = collect_fringe_nodes();
    cout << "Size of fringe nodes: " << fringe_nodes.size() << endl;
    current_node = nullptr;

    current_h_size = fringe_nodes.size();
    n_tested_nodes = 0;
}