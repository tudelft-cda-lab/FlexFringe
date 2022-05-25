#include <queue>
#include "searcher.h"
#include "parameters.h"

using namespace std;

/* queue used for searching */
struct refinement_list_compare{ bool operator()(const pair<double, refinement_list*> &a, const pair<double, refinement_list*> &b) const{ return a.first > b.first; } };
priority_queue< pair<double, refinement_list*>, vector< pair<double, refinement_list*> >, refinement_list_compare> Q;
refinement_list* current_refinements;
refinement_list* current_run;

double best_solution = -1;
int start_size = 0;
int nr = 0;

int greedy(state_merger* merger, int depth, bool undo){
    int result = depth;

    refinement_list refs;
    refinement* top_ref;
    if(current_run->empty()){
        top_ref = merger->get_best_refinement();
    } else {
        top_ref = current_run->front();
        current_run->pop_front();
        if(!top_ref->testref(merger)){
            top_ref->erase();
            top_ref = merger->get_best_refinement();
        }
    }
    while(top_ref != nullptr){
        refs.push_front(top_ref);
        nr++;

        top_ref->doref(merger);

        result++;
        if(current_run->empty()){
            top_ref = merger->get_best_refinement();
        } else {
            top_ref = current_run->front();
            current_run->pop_front();
            if(!top_ref->testref(merger)){
                top_ref->erase();
                top_ref = merger->get_best_refinement();
            }
        }
    }

    current_run->clear();

    result = merger->get_eval()->compute_global_score(merger);

    if(best_solution == -1.0 || result < best_solution){
        cerr << "*** current best *** " << result << endl;
        best_solution = result;
        merger->print_dot("search" + to_string(result) + ".dot");
        merger->print_json("search" + to_string(result) + ".json");
    }

    if(undo){
        for(refinement_list::iterator it = refs.begin(); it != refs.end(); ++it){
            current_run->push_front(*it);
            nr++;
            (*it)->undo(merger);
        }
    }
    return result;
}

double compute_score(state_merger* merger){
    if(SEARCH_DEEP) return greedy(merger, current_refinements->size(), true);
    if(SEARCH_LOCAL) return merger->get_best_refinement_score();
    if(SEARCH_GLOBAL) return merger->get_eval()->compute_global_score(merger);
    if(SEARCH_PARTIAL) return merger->get_eval()->compute_partial_score(merger);
    return current_refinements->size();
}

void add_to_q(state_merger* merger){
    double result = merger->get_eval()->compute_partial_score(merger);
    if(best_solution != -1.0 && result >= best_solution){
        cerr << "solution > best q_size: " << Q.size() << endl;
        return;
    }

    refinement_set* refs = merger->get_possible_refinements();

    if(refs->empty()){
        cerr << "solution: " << result << " q_size: " << Q.size() << endl;

        if(best_solution == -1.0 || result < best_solution){
            cerr << "*** current best *** " << result << endl;
            best_solution = result;
            merger->print_dot("search" + to_string(result) + ".dot");
            merger->print_json("search" + to_string(result) + ".json");
        }

        delete refs;
        return;
    }

    //refinement* top_ref = *refs->begin();

	for(refinement_set::iterator it = refs->begin(); it != refs->end(); ++it){
        refinement* ref = *it;
	    ref->doref(merger);
		double score = compute_score(merger);
        ref->undo(merger);

        refinement_list* new_list = new refinement_list(*current_refinements);

        for(refinement_list::iterator new_it = new_list->begin(); new_it != new_list->end(); new_it++){
            (*new_it)->increfs();
        }
        new_list->push_back(ref);
		Q.push(pair<double, refinement_list*>(score, new_list));
	}
	delete refs;
}

void change_refinement_list(state_merger* merger, refinement_list* new_list){
	refinement_list::reverse_iterator old_it = current_refinements->rbegin();
	while(old_it != current_refinements->rend()){

        (*old_it)->undo(merger);
        (*old_it)->erase();
		old_it++;
	}
	refinement_list::iterator new_it = new_list->begin();
	while(new_it != new_list->end()){
		(*new_it)->doref(merger);
		new_it++;
	}

	delete current_refinements;
	current_refinements = new_list;
}

void bestfirst(state_merger* merger){
	start_size = merger->get_final_apta_size();
    best_solution = -1;
    current_run = new refinement_list();

    current_refinements = new refinement_list();

    if(SEARCH_SINKS){
        int temp_merge_sinks = MERGE_SINKS;
        int temp_merge_sinks_core = MERGE_SINKS_WITH_CORE;
        MERGE_SINKS = 0;
        MERGE_SINKS_WITH_CORE = 0;
        greedy(merger, 0, false);
        best_solution = -1;
        MERGE_SINKS = temp_merge_sinks;
        MERGE_SINKS_WITH_CORE = temp_merge_sinks_core;
    }

	add_to_q(merger);

    cerr << Q.size() << endl;
	
	while(!Q.empty()){
		pair<double, refinement_list*> next_refinements = Q.top();
		change_refinement_list(merger, next_refinements.second);
		Q.pop();

		cerr << Q.size() << " " << current_refinements->size() << " " << next_refinements.first << endl;

        add_to_q(merger);
	}
}
