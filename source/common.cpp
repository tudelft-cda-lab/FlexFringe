/**
 * @file common.cpp
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "common.h"

#include <vector>
#include <string>

#include <iostream>

void print_current_automaton(state_merger* merger, const std::string& output_file, const std::string& append_string){
    if (OUTPUT_TYPE == "dot" || OUTPUT_TYPE == "both") {
        merger->print_dot(output_file + append_string + ".dot");
    }
    if (OUTPUT_TYPE == "json" || OUTPUT_TYPE == "both") {
        merger->print_json(output_file + append_string + ".json");
    }
    if(OUTPUT_SINKS && !PRINT_WHITE){
        bool red_undo = PRINT_RED;
        PRINT_RED = false;
        bool white_undo = PRINT_WHITE;
        PRINT_WHITE= true;
        bool blue_undo = PRINT_BLUE;
        PRINT_BLUE = true;
        if (OUTPUT_TYPE == "dot" || OUTPUT_TYPE == "both") {
            merger->print_dot(output_file + append_string + "sinks.dot");
        }
        if (OUTPUT_TYPE == "json" || OUTPUT_TYPE == "both") {
            merger->print_json(output_file + append_string + "sinks.json");
        }
        PRINT_RED = red_undo;
        PRINT_WHITE = white_undo;
        PRINT_BLUE = blue_undo;
    }
}

evaluation_function* get_evaluation(){
    std::cout << "Using evaluation class " << HEURISTIC_NAME << std::endl;

    evaluation_function *eval = nullptr;
    if(debugging_enabled){
        for(auto & myit : *DerivedRegister<evaluation_function>::getMap()) {
            std::cout << myit.first << " " << myit.second << std::endl;
        }
    }
    try {
        eval = (DerivedRegister<evaluation_function>::getMap())->at(HEURISTIC_NAME)();
        std::cerr << "Using heuristic " << HEURISTIC_NAME << std::endl;
        LOG_S(INFO) <<  "Using heuristic " << HEURISTIC_NAME;
    } catch(const std::out_of_range& oor ) {
        LOG_S(WARNING) << "No named heuristic found, defaulting back on -h flag";
        std::cerr << "No named heuristic found, defaulting back on -h flag" << std::endl;
    }
    return eval;
}

double update_score(double old_score, apta_node* next_node, tail* next_tail){
    double score = compute_score(next_node, next_tail);
    if(PREDICT_MINIMUM) return std::min(old_score, score);
    return old_score + score;
}

apta_node* single_step(apta_node* n, tail* t, apta* a){
    apta_node* child = n->child(t);
    if(child == 0){
        if(PREDICT_RESET) return a->get_root();
        else if(PREDICT_REMAIN) return n;
        else return nullptr;
    }
    return child->find();
}

double compute_score(apta_node* next_node, tail* next_tail){
    //if(PREDICT_ALIGN){ cerr << next_node->get_data()->align_score(next_tail) << endl; }
    if(PREDICT_ALIGN){ return next_node->get_data()->align_score(next_tail); }
    return next_node->get_data()->predict_score(next_tail);
}