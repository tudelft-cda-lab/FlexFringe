/**
 * @file interactive_mode.cpp
 * @author Christian Hammerschmidt (hammerschmidt@posteo.de)
 * @brief 
 * @version 0.1
 * @date 2024-12-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "interactive_mode.h"
#include "refinement.h"
#include "parameters.h"

#include <sstream>

void interactive_mode::initialize() {
    running_mode_base::initialize();
    read_input_file();
    
    eval->initialize_before_adding_traces();
    id.add_traces_to_apta(the_apta);
    eval->initialize_after_adding_traces(merger);
}

void interactive_mode::generate_output(){
    std::cout << "Printing output to " << OUTPUT_FILE << ".final" << std::endl;
    print_current_automaton(merger, OUTPUT_FILE, ".final");
}

/*! @brief Main loop for interactive mode.
 *         
 *  Constructs batch APTA, gets and prints possible merges, prompts user for interaction, executes command.
 *  Loops until some terminal condition is reached or user terminates the session.
 *  Outputs into two dot files to visualize current step/last step (merged) APTA.
 * @param[in] merger state_merger* object
 * @param[in] param  parameters* object to set global variables
 */
void interactive_mode::run(){
    std::cerr << "starting greedy merging" << std::endl; // cerr?
    int num = 1;
    refinement_list* all_refs = new refinement_list();
    merger->get_eval()->initialize_after_adding_traces(merger);

    std::string command = std::string("");
    std::string arg;
    int choice = 1; // what merge was picked
    int pos = 0; // what merge we are looking at
    int step = 1; // current step number
    int countdown = 0; // where to get to

    // if there is only one choice, do we auto-execute
    // or read for confirmation & chance for parameter changes?
    bool manual = false;
    bool execute = false;

    refinement_set* refs;
    refinement* chosen_ref;

    while( true ){
        while( true ){
            std::cout << " ";
            // output current merged apta
            merger->print_dot("pre_" + std::to_string(num % 2) + ".dot");

            refs = merger->get_possible_refinements();
            chosen_ref = *refs->begin();

            std::cout << "Possible refinements: " << std::endl;
            for(refinement_set::iterator it = refs->begin(); it != refs->end(); ++it){
                (*it)->print();
                std::cout << " , ";
            }

            std::cout << std::endl << "Your choice at step " << step << ": " << std::endl;
            getline(std::cin, command);

            std::stringstream cline(command);
            cline >> arg;

            // parse (prefix-free) commands
            if(arg == "undo_merge") {
                // undo_merge single merge
                (*all_refs->begin())->undo(merger);
                all_refs->pop_front();
                step--;
                std::cout << "undid last merge" << std::endl;
            } else if(arg == "restart") {
                // undo_merge all merges up to here
                while(all_refs->begin() != all_refs->end()) {
                    (*all_refs->begin())->undo(merger);
                    all_refs->pop_front();
                }
                step = 1;
                std::cout << "Restarted" << std::endl;
            } else if (arg == "leap") {
                // automatically do the next n steps
                // TODO: error handling for n = NaN
                cline >> arg;
                countdown = stoi(arg);

                manual = true;
                arg = std::string("1");
                execute = true;
                break;
            } else if(arg == "set") {
                // set PARAM <value>, call init_with_param
                cline >> arg;
                if(arg == "state_count") {
                    cline >> arg;
                    STATE_COUNT = stoi(arg);
                    std::cout << "STATE_COUNT is now " << STATE_COUNT << std::endl;
                }
                if(arg == "symbol_count") {
                    cline >> arg;
                    SYMBOL_COUNT = stoi(arg);
                    std::cout << "SYMBOL_COUNT is now " << SYMBOL_COUNT << std::endl;
                }
                if(arg == "lower_bound") {
                    cline >> arg;
                    LOWER_BOUND = stoi(arg);
                    std::cout << "LOWER_BOUND is now " << LOWER_BOUND << std::endl;
                }
                if(arg == "sinkson") {
                    cline >> arg;
                    USE_SINKS = stoi(arg);
                    std::cout << "USE_SINKS is now " << (USE_SINKS==true ? "true" : "false") << std::endl;
                }
                if(arg == "blueblue") {
                    cline >> arg;
                    MERGE_BLUE_BLUE = stoi(arg);
                    std::cout << "MERGE_BLUE_BLUE is now " << (MERGE_BLUE_BLUE==true ? "true" : "false") << std::endl;
                }
                if(arg == "shallowfirst") {
                    cline >> arg;
                    DEPTH_FIRST = stoi(arg);
                    std::cout << "SHALLOW_FIRST is now " << (DEPTH_FIRST==true ? "true" : "false") << std::endl;
                }
                if(arg == "largestblue") {
                    cline >> arg;
                    MERGE_MOST_VISITED = stoi(arg);
                    std::cout << "MERGE_MOST_VISITED is now " << (MERGE_MOST_VISITED==true ? "true" : "false") << std::endl;
                }
            } else if(arg == "force") {
                // implements are mandatory merge
                std::cout << "State two sequences in abg format, ending in the same state: " << std::endl;
                std::string seq1 = "";
                std::string seq2 = "";


            } else if(arg == "help") {
                std::cout << "Available commands: set <param> value, undo_merge, help; insert <sample> in abd format; <int> merges the <int>th merge from the proposed list" << std::endl;
                // next command?

            } else {
                try {
                    choice = stoi(arg);
                    break;
                } catch(std::invalid_argument e) {
                    std::cout << "Invalid command. Try \"help\" if you are lost" << std::endl;
                    execute = false;
                }
            }
        }
        // track number of ops on APTA
        step++;

        // auto-execute/leap steps
        if(countdown == 1) {
            manual = false;
            step = 1;
        }
        if(countdown > 0) {
            countdown--;
        }

        // find chosen refinement and execute
        for(refinement_set::iterator it = refs->begin(); it != refs->end(); ++it){
            if(pos == choice) chosen_ref = *it;
            pos++;
        }
        pos = 1;

        // execute choice
        if(refs->empty()){
            std::cerr << "no more possible merges" << std::endl;
            break;
        }

        // chosen ref instead of best ref
        chosen_ref->print_short();
        std::cerr << " ";

        chosen_ref->doref(merger);

        all_refs->push_front(chosen_ref);

        for(refinement_set::iterator it = refs->begin(); it != refs->end(); ++it){
            if(*it != chosen_ref) (*it)->erase();
        }
        delete refs;
        num = num + 1;

        execute = false;
    }
    std::cout << std::endl;

    int size =  merger->get_final_apta_size();
    int red_size = merger->get_num_red_states();
    std::cout << std::endl << "Found heuristic solution with " << size << " states, of which " << red_size << " are red states." << std::endl;
};


