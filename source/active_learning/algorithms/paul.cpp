/**
 * @file paul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-08-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "paul.h"
#include "base_teacher.h"
#include "common_functions.h"
#include "input_file_oracle.h"
#include "input_file_sul.h"

#include "greedy.h"
#include "inputdata.h"
#include "main_helpers.h"
#include "mem_store.h"
#include "parameters.h"
#include "state_merger.h"

#include <list>

using namespace std;
using namespace active_learning_namespace;

refinement* paul_algorithm::get_best_refinement(unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta, unique_ptr<base_teacher>& teacher){
    state_set blue_its = state_set();
    //bool found_non_sink = false;
    
    for (blue_state_iterator it = blue_state_iterator(the_apta->get_root()); *it != nullptr; ++it){
        auto blue_node = *it;
        if(blue_node->get_size() != 0) blue_its.insert(blue_node);
        //if(sink_type(blue_node) == -1) found_non_sink = true;
    }

    //if (!found_non_sink && !MERGE_SINKS){
    //    return result;
    //}

    auto rs = make_unique<refinement_set>();
    for (auto blue_node: blue_its) {
        bool mergeable = false;
        for(red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it){
            auto red_node = *r_it;
            ii_handler->complement_nodes(the_apta, teacher, red_node, blue_node, 0);

            refinement* ref = merger->test_merge(red_node, blue_node);
            if (ref != nullptr){
                rs->insert(ref);
                mergeable = true;
            }
        }
        if(!mergeable)
            rs->insert(mem_store::create_extend_refinement(merger.get(), blue_node));
    }

    refinement *r = nullptr;
    if (!rs->empty()) {
        r = *(rs->begin());
        for(auto it = rs->begin(); it != rs->end(); ++it){
            auto rf = *it;
            if(r != rf) rf->erase();
        }
    }

    return r;
}


void paul_algorithm::run(inputdata& id) {
    int n_runs = 1;

#ifndef NDEBUG
    LOG_S(INFO) << "Creating debug directory " << DEBUG_DIR;
    std::filesystem::create_directories(DEBUG_DIR);
#endif
    LOG_S(INFO) << "Running PAUL algorithm.";

    auto eval = unique_ptr<evaluation_function>(get_evaluation());
    eval->initialize_before_adding_traces();

    auto the_apta = unique_ptr<apta>(new apta());
    auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));
    this->oracle->initialize(merger.get());
    the_apta->set_context(merger.get());
    eval->set_context(merger.get());
    
    id.add_traces_to_apta(the_apta.get());
    eval->initialize_after_adding_traces(merger.get());

    const vector<int> alphabet = id.get_alphabet();
    cout << "Alphabet: ";
    active_learning_namespace::print_sequence<vector<int>::const_iterator>(alphabet.cbegin(), alphabet.cend());

    auto* best_ref = paul_algorithm::get_best_refinement(merger, the_apta, teacher);
    int num = 0;
    while(best_ref != nullptr){
        cout << " ";
        best_ref->print_short();
        cout << " ";
        std::cout.flush();

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << num;
        std::string s = ss.str();

#ifndef NDEBUG
        merger->print_dot("test" + ss.str() + ".dot");
#endif

        best_ref->doref(merger.get());

        delete best_ref;
        best_ref = paul_algorithm::get_best_refinement(merger, the_apta, teacher);

        num++;
    }

    cout << "Found automaton. Printing and terminating." << endl;
    print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
}
