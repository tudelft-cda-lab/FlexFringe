/// @file interactive.cpp
/// @brief All the functions and definitions for interactive merge mode.
/// @author Christian Hammerschmidt, hammerschmidt@posteo.de

#include <sstream>
#include "refinement.h"
#include "greedy.h"
#include "parameters.h"

/*! @brief Main loop for interactive mode.
 *         
 *  Constructs batch APTA, gets and prints possible merges, prompts user for interaction, executes command.
 *  Loops until some terminal condition is reached or user terminates the session.
 *  Outputs into two dot files to visualize current step/last step (merged) APTA.
 * @param[in] merger state_merger* object
 * @param[in] param  parameters* object to set global variables
 * @return refinement_list* list of refinments executed by the state merger
 */
refinement_list* interactive(state_merger* merger){
    cerr << "starting greedy merging" << endl; // cerr?
    int num = 1;
    refinement_list* all_refs = new refinement_list();
    merger->get_eval()->initialize_after_adding_traces(merger);

    string command = string("");
    string arg;
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
            cout << " ";
            // output current merged apta
            merger->print_dot("pre_" + to_string(num % 2) + ".dot");

            refs = merger->get_possible_refinements();
            chosen_ref = *refs->begin();

            cout << "Possible refinements: " << endl;
            for(refinement_set::iterator it = refs->begin(); it != refs->end(); ++it){
                (*it)->print();
                cout << " , ";
            }

            cout << endl << "Your choice at step " << step << ": " << endl;
            getline(std::cin, command);

            stringstream cline(command);
            cline >> arg;

            // parse (prefix-free) commands
            if(arg == "undo_merge") {
                // undo_merge single merge
                (*all_refs->begin())->undo(merger);
                all_refs->pop_front();
                step--;
                cout << "undid last merge" << endl;
            } else if(arg == "restart") {
                // undo_merge all merges up to here
                while(all_refs->begin() != all_refs->end()) {
                    (*all_refs->begin())->undo(merger);
                    all_refs->pop_front();
                }
                step = 1;
                cout << "Restarted" << endl;
            } else if (arg == "leap") {
                // automatically do the next n steps
                // TODO: error handling for n = NaN
                cline >> arg;
                countdown = stoi(arg);

                manual = true;
                arg = string("1");
                execute = true;
                break;
            } else if(arg == "set") {
                // set PARAM <value>, call init_with_param
                cline >> arg;
                if(arg == "state_count") {
                    cline >> arg;
                    STATE_COUNT = stoi(arg);
                    cout << "STATE_COUNT is now " << STATE_COUNT << endl;
                }
                if(arg == "symbol_count") {
                    cline >> arg;
                    SYMBOL_COUNT = stoi(arg);
                    cout << "SYMBOL_COUNT is now " << SYMBOL_COUNT << endl;
                }
                if(arg == "lower_bound") {
                    cline >> arg;
                    LOWER_BOUND = stoi(arg);
                    cout << "LOWER_BOUND is now " << LOWER_BOUND << endl;
                }
                if(arg == "sinkson") {
                    cline >> arg;
                    USE_SINKS = stoi(arg);
                    cout << "USE_SINKS is now " << (USE_SINKS==true ? "true" : "false") << endl;
                }
                if(arg == "blueblue") {
                    cline >> arg;
                    MERGE_BLUE_BLUE = stoi(arg);
                    cout << "MERGE_BLUE_BLUE is now " << (MERGE_BLUE_BLUE==true ? "true" : "false") << endl;
                }
                if(arg == "shallowfirst") {
                    cline >> arg;
                    DEPTH_FIRST = stoi(arg);
                    cout << "SHALLOW_FIRST is now " << (DEPTH_FIRST==true ? "true" : "false") << endl;
                }
                if(arg == "largestblue") {
                    cline >> arg;
                    MERGE_MOST_VISITED = stoi(arg);
                    cout << "MERGE_MOST_VISITED is now " << (MERGE_MOST_VISITED==true ? "true" : "false") << endl;
                }
            } else if(arg == "force") {
                // implements are mandatory merge
                cout << "State two sequences in abg format, ending in the same state: " << endl;
                string seq1 = "";
                string seq2 = "";


            } else if(arg == "help") {
                cout << "Available commands: set <param> value, undo_merge, help; insert <sample> in abd format; <int> merges the <int>th merge from the proposed list" << endl;
                // next command?

            } else {
                try {
                    choice = stoi(arg);
                    break;
                } catch(std::invalid_argument e) {
                    cout << "Invalid command. Try \"help\" if you are lost" << endl;
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
            cerr << "no more possible merges" << endl;
            break;
        }

        // chosen ref instead of best ref
        chosen_ref->print_short();
        cerr << " ";

        chosen_ref->doref(merger);

        all_refs->push_front(chosen_ref);

        for(refinement_set::iterator it = refs->begin(); it != refs->end(); ++it){
            if(*it != chosen_ref) (*it)->erase();
        }
        delete refs;
        num = num + 1;

        execute = false;
    }
    cout << endl;

    int size =  merger->get_final_apta_size();
    int red_size = merger->get_num_red_states();
    cout << endl << "Found heuristic solution with " << size << " states, of which " << red_size << " are red states." << endl;
    return all_refs;
};


