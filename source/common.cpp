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