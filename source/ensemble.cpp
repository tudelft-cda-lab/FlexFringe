#include <stdio.h>
#include <sstream>
#include <fstream>
#include <cstdlib>

#include "refinement.h"
#include "greedy.h"
#include "parameters.h"

/** todo: work in progress */

refinement_list* greedy(state_merger* merger){
    cerr << "starting greedy merging" << endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    auto* all_refs = new refinement_list();

    refinement* best_ref = merger->get_best_refinement();
    while( best_ref != nullptr ){
        cout << " ";
        best_ref->print_short();
        cout << " ";
        std::cout.flush();

        best_ref->doref(merger);
        all_refs->push_back(best_ref);
        best_ref = merger->get_best_refinement();
    }
    cout << "no more possible merges" << endl;
    return all_refs;
};

void bagging(state_merger* merger, string output_file, int nr_estimators){
    cerr << "starting bagging" << endl;
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
    cerr << "ended bagging" << endl;
};
