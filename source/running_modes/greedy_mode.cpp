/**
 * @file greedy_mode.cpp
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "greedy_mode.h"
#include "common.h"
#include "refinement.h"
#include "parameters.h"
#include "output_manager.h"

#include <sstream>
#include <iostream>

using namespace std;

void greedy_mode::initialize(){
    running_mode_base::initialize();
    read_input_file();
    
    eval->initialize_before_adding_traces();
    id.add_traces_to_apta(the_apta);
    eval->initialize_after_adding_traces(merger);

    cout << "Printing initial tree to " << OUTPUT_FILE << ".init" << endl;
    output_manager::print_current_automaton(merger, OUTPUT_FILE, ".init");
}

int greedy_mode::run(){
    std::cerr << "Starting greedy merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    //auto* all_refs = new refinement_list();

    refinement* best_ref = merger->get_best_refinement();
    int num = 1;

    while( best_ref != nullptr ){
        std::cout << " ";
        best_ref->print_short();
        std::cout << " ";
        std::cout.flush();

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << num;
        std::string s = ss.str();

        if (DEBUGGING) {
            merger->print_dot("test" + ss.str() + ".dot");
        }

        best_ref->doref(merger);

        //all_refs->push_back(best_ref);
        delete best_ref;
        best_ref = merger->get_best_refinement();

        num++;
    }

    std::cout << "no more possible merges" << std::endl;
    return EXIT_SUCCESS;
}