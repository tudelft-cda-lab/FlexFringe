/**
 * @file targeted_bfs_walk.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-05-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "targeted_bfs_walk.h"
#include "parameters.h"

#include <cassert>
#include <iostream>
#include <list>
#include <vector>

using namespace std;

int targeted_bfs_walk::count_nodes(){
    int res = 0;
    for (red_state_iterator current_node_it = red_state_iterator(hypothesis->get_root()); *current_node_it != nullptr; ++current_node_it)
        ++res;
    
    return res;
}

void targeted_bfs_walk::add_nodes_to_queue(apta_node* node){
    static const auto mu = MU;

    auto data = node->get_data(); 
    for(int symbol: this->alphabet){
        if(data->get_weight(symbol) == 0){
            ++n_tested_nodes;
            continue;
        }

        auto next_node = current_node->get_child(symbol);
        if(queued_nodes.contains(next_node)) continue;
        
        node_queue.push(next_node);
        queued_nodes.insert(next_node);
    }
}


optional<vector<int>> targeted_bfs_walk::next(const inputdata& id) {
    while(tested_nodes.contains(current_node)){
        if(node_queue.empty()) return nullopt; // we tested all nodes

        add_nodes_to_queue(current_node);
        current_node = node_queue.front();
        node_queue.pop();
    }

    if(samples_for_current_node == SAMPLES_PER_NODE){
        tested_nodes.insert(current_node);

        if(node_queue.empty()) return nullopt; // we tested all nodes

        current_node = node_queue.front();
        add_nodes_to_queue(current_node);
        node_queue.pop();

        auto at = current_node->get_access_trace();
        current_access_sequence = at->get_input_sequence(true, false);
        
        cout << "Shifting to access trace: ";
        for(int x: current_access_sequence) 
            cout << x << " ";
        cout << endl;

        samples_for_current_node = 0;
        ++n_tested_nodes;
    }

    if(samples_for_current_node % 1000 == 0)
        cout << "Processed nodes:" << n_tested_nodes << "/" << current_h_size << ". Samples: " << samples_for_current_node << endl;

    const int output_string_length = length_generator.get_random_int();
    vector<int> res(current_access_sequence);
    for (int i = 0; i < output_string_length; ++i) 
        res.push_back(alphabet[alphabet_sampler.get_random_int()]);

    ++samples_for_current_node;
    return res;
}

void targeted_bfs_walk::initialize(state_merger* merger){
    this->hypothesis = merger->get_aut();
    this->alphabet = merger->get_dat()->get_alphabet();

    n_tested_nodes = 0;
    alphabet_sampler.set_limits(0, alphabet.size() - 1);
}

void targeted_bfs_walk::reset(){
    if(hypothesis==nullptr){
      cerr << "ERROR: Tried to call reset() method in targeted_bfs_walk without initializing the hypothesis (==nullptr). Aborting program." << std::endl;
      throw exception();
    }
    
    // reset the structures
    node_queue = queue<apta_node*>();
    queued_nodes.clear();

    auto root_node_it = red_state_iterator(hypothesis->get_root());
    current_node = *root_node_it; 
    samples_for_current_node = 0;
    add_nodes_to_queue(current_node);

    auto at = current_node->get_access_trace();
    current_access_sequence = at->get_input_sequence(true, false);

    current_h_size = count_nodes();
}