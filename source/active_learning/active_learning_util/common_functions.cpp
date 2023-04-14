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
#include "inputdata.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>

using namespace std;

//const bool PRINT_ALL_MODELS = false;

apta_node* active_learning_namespace::get_child_node(apta_node* n, tail* t){
    apta_node* child = n->child(t);
    if(child == 0){
        return nullptr;
    }
    return child->find();
}

/**
 * @brief There are two versions of this function. In this version we look at if the tree is 
 * possibly parsable by the traces. 
 * 
 * The problem with this one is that in algorithms where we expect positive and negative traces, 
 * then the resulting DFA will be the root node with lots of self-loops only. Reason: This way we
 * won't get a counterexample, all traces will be accepted.
 * 
 * @param tr The trace.
 * @param aut The apta.
 * @return true Accepts trace.
 * @return false Does not accept trace.
 */
bool active_learning_namespace::aut_accepts_trace(trace* tr, apta* aut){
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
 * @brief Predicts type from trace. Code is a subset of the predict.cpp functions, hence duplicate code. TODO: clean that up
 * 
 * @param tr The trace.
 * @return const int The predicted type.
 */
const int active_learning_namespace::predict_type_from_trace(trace* tr, apta* aut, inputdata& id){
    apta_node* n = aut->get_root();
    tail* t = tr->get_head(); // TODO: get
    for(int j = 0; j < t->get_length(); j++){
        n = get_child_node(n, t);
        if(n == nullptr){
            return id.get_reverse_type(REJECTING_LABEL);
        }
        t = t->future();
    }

    apta_node* ending_state = n;
    tail* ending_tail = t;
    if(ending_tail->is_final()) ending_tail = ending_tail->past();

    return ending_state->get_data()->predict_type(ending_tail);
}

/**
 * @brief This is the other version of the function. This one uses the types of the traces, 
 * i.e. it implements accepting and rejecting traces. Hence we get a different case, and this is 
 * the version that we use for e.g. L* and L#.
 * 
 * @param tr The trace.
 * @param eval The evaluation function. Must inherit from count_driven.
 * @return true Accepts trace.
 * @return false Does not.
 */
bool active_learning_namespace::aut_accepts_trace(trace* tr, apta* aut, const count_driven* const eval){
    const int trace_type = tr->get_type(); //eval->predict_type(tr);
    
    apta_node* n = aut->get_root();
    tail* t = tr->get_head();
    for(int j = 0; j < t->get_length(); j++){
        n = active_learning_namespace::get_child_node(n, t);
        if(n == nullptr) return false; // TODO: does this one make sense?

        t = t->future();
    }

    if(trace_type==n->get_data()->predict_type(t)) return true;
    return false;
}


/**
 * @brief This function is like the greedyrun method, but it additionally returns the refinements done. 
 * We need this in active learning that whenever the equivalence oracle does not work out we will be able
 * to undo all the refinments and pose a fresh hypothesis later on.
 * 
 * @param aut The apta.
 * @return list<refinement*> list with the refinements done. 
 */
const list<refinement*> active_learning_namespace::minimize_apta(state_merger* merger){
    list<refinement*> refs;
    refinement* top_ref = merger->get_best_refinement();

    while(top_ref != 0){
        refs.push_back(top_ref);        
        top_ref->doref(merger);
        top_ref = merger->get_best_refinement();
    }
    return refs;
}

/**
 * @brief Resets the apta.
 * 
 * Side effect: Exhausts the refs-stack to zero.
 * 
 * @param merger The state merger.
 * @param refs Stack with refinements.
 */
void active_learning_namespace::reset_apta(state_merger* merger, const list<refinement*>& refs){
    for(auto it = refs.rbegin(); it != refs.rend(); ++it){
        const auto top_ref = *it;
        top_ref->undo(merger);
    }
}

void active_learning_namespace::update_tail(tail* t, const int symbol){
    static int num_tails = 0;

    tail_data* td = t->td;
    td->symbol = symbol;
    //td->data = ""; // TODO: does not work yet with attributes
    td->tail_nr = num_tails++;

    int num_symbol_attributes = 0; //inputdata::get_num_symbol_attributes();
    if(num_symbol_attributes > 0){
      // TODO: we do not treat this one yet
    }
}

/**
 * @brief Add the sequence as a concatenation of tail-objects to the trace, so that flexfringe can work it out.
 * 
 * @param new_trace The trace to add to.
 * @param sequence Sequence in list for.
 */
void active_learning_namespace::add_sequence_to_trace(trace* new_trace, const list<int> sequence){
    new_trace->length = sequence.size();
    
    tail* new_tail = mem_store::create_tail(nullptr);
    new_tail->tr = new_trace;
    new_trace->head = new_tail;

    auto sequence_it = sequence.cbegin();
    for(int index = 0; index < sequence.size(); ++index){
        const int symbol = *sequence_it;
        active_learning_namespace::update_tail(new_tail, symbol);
        new_tail->td->index = index;

        tail* old_tail = new_tail;
        new_tail = mem_store::create_tail(nullptr);
        new_tail->tr = new_trace;
        old_tail->set_future(new_tail);

        ++sequence_it;
    }

    new_tail->td->index = sequence.size();
    new_trace->end_tail = new_tail;

    new_trace->finalize();
}

/**
 * @brief What you think it does.
 * 
 */
list<int> active_learning_namespace::concatenate_strings(const list<int>& pref1, const list<int>& pref2){
  list<int> res(pref1);
  res.insert(res.end(), pref2.begin(), pref2.end());
  return res;
}

/**
 * @brief Turns a list to a trace.
 * 
 * @param vec The list.
 * @param id The inputdata.
 * @param trace_type Accepting or rejecting.
 * @return trace* The trace.
 */
trace* active_learning_namespace::vector_to_trace(const list<int>& vec, inputdata& id, const int trace_type){
    static int trace_nr = 0;

    trace* new_trace = mem_store::create_trace(&id);

    new_trace->type = trace_type;
    new_trace->sequence = ++trace_nr;

    active_learning_namespace::add_sequence_to_trace(new_trace, vec);
    
    return new_trace;
}

/**
 * @brief Print a list of ints. For debugging purposes.
 * 
 * @param v The list.
 */
template<class it_T>
void active_learning_namespace::print_sequence(it_T begin, it_T end){
    cout << "seq: ";
    for (; begin != end; ++begin)
        cout << *begin << " ";
    cout << endl;
}