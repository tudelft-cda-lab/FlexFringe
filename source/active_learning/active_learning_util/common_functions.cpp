/**
 * @file common_functions.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "common_functions.h"
#include "parameters.h"
#include "definitions.h"

#include <iostream>
#include <stdexcept>

using namespace std;
using namespace active_learning_namespace;

/* evaluation_function* get_evaluation(){
    evaluation_function *eval = nullptr;
    if(debugging_enabled){
        for(auto & myit : *DerivedRegister<evaluation_function>::getMap()) {
            cout << myit.first << " " << myit.second << endl;
        }
    }
    try {
        eval = (DerivedRegister<evaluation_function>::getMap())->at(HEURISTIC_NAME)();
        std::cout << "Using heuristic " << HEURISTIC_NAME << std::endl;
        LOG_S(INFO) <<  "Using heuristic " << HEURISTIC_NAME;
    } catch(const std::out_of_range& oor ) {
        LOG_S(WARNING) << "No named heuristic found, defaulting back on -h flag";
        std::cerr << "No named heuristic found, defaulting back on -h flag" << std::endl;
    }
    return eval;
} */

apta_node* active_learning_namespace::get_child_node(apta_node* n, tail* t){
    apta_node* child = n->child(t);
    if(child == 0){
        return nullptr;
    }
    return child->find();
}

bool aut_accepts_trace(trace* tr, apta* aut){
    apta_node* n = aut->get_root();
    tail* t = tr->get_head();
    for(int j = 0; j < t->get_length(); j++){
        n = active_learning_namespace::get_child_node(n, t);
        if(n == nullptr) return false; // TODO: does this one make sense?

        t = t->future();
    }
    return true;
}

/**
 * @brief This function is like the greedyrun method, but it additionally returns the refinements done. 
 * We need this in active learning that whenever the equivalence oracle does not work out we will be able
 * to undo all the refinments and pose a fresh hypothesis later on.
 * 
 * @param aut The apta.
 * @return vector<refinement*> vector with the refinements done. 
 */
const vector<refinement*> minimize_apta(state_merger* merger){
    vector<refinement*> refs;

    refinement* top_ref = merger->get_best_refinement();
    while(top_ref != 0){
        refs.push_back(top_ref);
        top_ref->doref(merger);
        top_ref = merger->get_best_refinement();
    }
    return refs;
}

void reset_apta(state_merger* merger, const vector<refinement*> refs){
    for(auto top_ref: refs){
        top_ref->undo(merger);
    }
}

void update_tail(tail* t, const int symbol){
    static int num_tails = 0;

    tail_data* td = new_tail->td;
    td->symbol = symbol;
    td->data = data;
    td->tail_nr = num_tails++;

    int num_symbol_attributes = inputdata::get_num_symbol_attributes();
    if(num_symbol_attributes > 0){
      // TODO: we do not treat this one yet
/*         l3.str(symbol_attr);
        for(int i = 0; i < num_symbol_attributes; ++i){
            if(i < num_symbol_attributes - 1) std::getline(l3,val,',');
            else std::getline(l3,val);
            td->attr[i] = symbol_attributes[i].get_value(val);
        }
        l3.clear(); */
    }
}

void add_sequence_to_trace(trace* new_trace, const vector<int> sequence){
    static int num_sequences = 0;
    new_trace->length = sequence.size();

    tail* new_tail = mem_store::create_tail(nullptr);
    new_tail->tr = new_trace;
    new_trace->head = new_tail;

    for(int index = 0; index < sequence.size(); ++index){
        const int symbol = sequence.at(index);
        update_tail(new_tail, symbol);

        new_tail->td->index = index;
        tail* old_tail = new_tail;
        new_tail = mem_store::create_tail(nullptr);
        new_tail->tr = new_trace;
        old_tail->set_future(new_tail);
    }
    new_tail->td->index = sequence.size();
    new_trace->end_tail = new_tail;
    new_trace->sequence = num_sequences++;
}

trace* vector_to_trace(const vector<int>& vec, const knowledge_t trace_type){
    trace* new_trace = mem_store::create_trace(id);
    int type;
    if(trace_type==knowledge_t::accepting){
      type = 1;
    }
    else if (trace_type==knowledge_t::rejecting){
      type = 0;
    }    
    else{
      throw logic_error("This part is not implemented (yet).");
    }
    new_trace->type = type;
    add_sequence_to_trace(new_trace, vec);
    new_trace->finalize();
    
    return new_trace;
}

