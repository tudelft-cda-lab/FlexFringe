/**
 * @file output_manager.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-31
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "output_manager.h"

#include <iostream>

using namespace std;

void output_manager::init_outfile_path(){
  if(OUTPUT_FILE.empty()) 
    outfile_path = INPUT_FILE + ".ff";
  else
    outfile_path = OUTPUT_FILE;

    // TODO: do something with the predict mode here after refactoring predict mode
}

string_view output_manager::get_outfile_path(){
  return outfile_path;
}

/**
 * @brief Prints the automaton. The output string is determined by the output parameter when running program (see init_outfile_path()).
 * 
 * Call this function normally to put out the final automaton. For all other outputs like intermediate outputs call the overloaded function.
 * 
 */
void output_manager::print_final_automaton(state_merger* merger, const std::string& append_string){
  print_current_automaton(merger, outfile_path, append_string);
}

/**
 * @brief For backwards compatibility and to allow other types of outputs as well, e.g. for debugging.
 */
void output_manager::print_current_automaton(state_merger* merger, const std::string& output_file, const std::string& append_string){
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