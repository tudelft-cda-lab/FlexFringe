/**
 * @file ensemble_mode.cpp
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "ensemble_mode.h"
#include "refinement.h"
#include "parameters.h"

#include <stdio.h>
#include <sstream>
#include <fstream>
#include <cstdlib>

using namespace std;

/** todo: work in progress */

void ensemble_mode::initialize(){
    throw invalid_argument("ensemble mode not finished implementing yet.");
    running_mode_base::initialize();
    read_input_file();

    eval->initialize_before_adding_traces();
    id.add_traces_to_apta(the_apta);
    eval->initialize_after_adding_traces(merger);
}

int ensemble_mode::run(){
    throw invalid_argument("ensemble mode not finished implementing yet.");
    // TODO: call bagging method
}

refinement_list* ensemble_mode::greedy(){
    std::cerr << "starting greedy merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    auto* all_refs = new refinement_list();

    refinement* best_ref = merger->get_best_refinement();
    while( best_ref != nullptr ){
        std::cout << " ";
        best_ref->print_short();
        std::cout << " ";
        std::cout.flush();

        best_ref->doref(merger);
        all_refs->push_back(best_ref);
        best_ref = merger->get_best_refinement();
    }
    std::cout << "no more possible merges" << std::endl;
    return all_refs;
};

void ensemble_mode::bagging(std::string output_file, int nr_estimators){
    std::cerr << "starting bagging" << std::endl;
    for(int i = 0; i < nr_estimators; ++i){
        refinement_list* all_refs = greedy(merger);

        for(refinement_list::reverse_iterator it = all_refs->rbegin(); it != all_refs->rend(); ++it){
            (*it)->undo(merger);
        }
        for(refinement_list::iterator it = all_refs->begin(); it != all_refs->end(); ++it){
            (*it)->erase();
        }
        delete all_refs;
    }
    std::cerr << "ended bagging" << std::endl;
};
