/**
 * @file random_w_method.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-05-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "random_w_method.h"

#include <cassert>
#include <iostream>
#include <list>
#include <vector>

using namespace std;

int random_w_method::count_nodes(){
    int res = 0;
    for (red_state_iterator r_it = red_state_iterator(hypothesis->get_root()); *r_it != nullptr; ++r_it)
        ++res;
    
    return res;
}

optional<vector<int>> random_w_method::next(const inputdata& id) {

    // TODO: we can move this block into new initialize method
    static bool initialized = false;
    if (!initialized) {
        alphabet_vec = id.get_alphabet();
        alphabet_sampler.set_limits(0, alphabet_vec.size() - 1);

        initialized = true;
    }

    while(tested_nodes.contains(*r_it))
        r_it.increment();

    if(samples_for_current_node == SAMPLES_PER_NODE){
        tested_nodes.insert(*r_it);

        r_it.increment();

        if(*r_it==nullptr) // we tested all nodes
            return nullopt;

        auto at = (*r_it)->get_access_trace();
        current_access_sequence = at->get_input_sequence(true, false);

        samples_for_current_node = 0;
        ++n_tested_nodes;
    }

    if(samples_for_current_node % 1000 == 0)
        cout << "Processed nodes:" << n_tested_nodes << "/" << current_h_size << ". Samples: " << samples_for_current_node << endl;

    const int output_string_length = length_generator.get_random_int();
    vector<int> res(current_access_sequence);
    for (int i = 0; i < output_string_length; ++i) 
        res.push_back(alphabet_vec[alphabet_sampler.get_random_int()]);

    ++samples_for_current_node;
    return res;
}

void random_w_method::initialize(state_merger* merger){
    this->hypothesis = merger->get_aut();

    r_it = red_state_iterator(hypothesis->get_root());
    samples_for_current_node = 0;

    auto at = (*r_it)->get_access_trace();
    current_access_sequence = at->get_input_sequence(true, false);

    n_tested_nodes = 0;
}

void random_w_method::reset(){
    if(hypothesis==nullptr){
      cerr << "ERROR: Tried to call reset() method in random_w_method without initializing the hypothesis (==nullptr). Aborting program." << std::endl;
      throw exception();
    }
        
    r_it = red_state_iterator(hypothesis->get_root());
    samples_for_current_node = 0;

    auto at = (*r_it)->get_access_trace();
    current_access_sequence = at->get_input_sequence(true, false);

    //n_tested_nodes = 0;

    current_h_size = count_nodes();
}