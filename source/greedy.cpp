
#include <sstream>
#include "refinement.h"
#include "greedy.h"
#include "parameters.h"

void greedy_run(state_merger* merger){
    cerr << "starting greedy merging" << endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    //auto* all_refs = new refinement_list();

    refinement* best_ref = merger->get_best_refinement();
    int num = 1;

    while( best_ref != nullptr ){
        cout << " ";
        best_ref->print_short();
        cout << " ";
        std::cout.flush();

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << num;
        std::string s = ss.str();

        if(DEBUGGING){
            merger->renumber_states();
            merger->print_dot("test" + ss.str() + ".dot");
        }

        best_ref->doref(merger);

        //all_refs->push_back(best_ref);
        delete best_ref;
        best_ref = merger->get_best_refinement();

        num++;
    }
    cout << "no more possible merges" << endl;
};



