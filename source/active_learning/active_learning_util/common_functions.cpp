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

apta_node* active_learning_namespace::get_child_node(apta_node* n, tail* t){
    apta_node* child = n->child(t);
    if(child == 0){
        return nullptr;
    }
    return child->find();
}

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
 * @brief This function is like the greedyrun method, but it additionally returns the refinements done. 
 * We need this in active learning that whenever the equivalence oracle does not work out we will be able
 * to undo all the refinments and pose a fresh hypothesis later on.
 * 
 * @param aut The apta.
 * @return vector<refinement*> vector with the refinements done. 
 */
const stack<refinement*> active_learning_namespace::minimize_apta(state_merger* merger){
    stack<refinement*> refs;
    refinement* top_ref = merger->get_best_refinement();
    while(top_ref != 0){
        refs.push(top_ref);        
        top_ref->doref(merger);
        top_ref = merger->get_best_refinement();
    }
    return refs;
}

void active_learning_namespace::reset_apta(state_merger* merger, stack<refinement*> refs){
    while(!refs.empty()){
        const auto& top_ref = refs.top();
        top_ref->undo(merger);
        refs.pop();
    }
}

void active_learning_namespace::update_tail(tail* t, const int symbol){
    static int num_tails = 0;

    tail_data* td = t->td;
    td->symbol = symbol;
    //td->data = ""; // TODO: does not work yet with attributes
    td->tail_nr = num_tails;
    ++num_tails;

    int num_symbol_attributes = 0; //inputdata::get_num_symbol_attributes();
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

/**
 * @brief Add the sequence as a concatenation of tail-objects to the trace, so that flexfringe can work it out.
 * 
 * @param new_trace The trace to add to.
 * @param sequence Sequence in vector for.
 */
void active_learning_namespace::add_sequence_to_trace(trace* new_trace, const vector<int> sequence){
    tail* new_tail = mem_store::create_tail(nullptr);
    new_tail->tr = new_trace;
    new_trace->head = new_tail;

    if(std::count(sequence.begin(), sequence.end(), active_learning_namespace::EPS) == sequence.size()) {
        new_tail->td->index = 0;
        
        new_trace->end_tail = new_tail;
        new_trace->length = 1;
        
        update_tail(new_tail, -1);
        new_trace->end_tail = new_tail;
        return; 
    } 

    int size = 0;
    for(int index = 0; index < sequence.size(); ++index){
        const int symbol = sequence.at(index);
        if(symbol==EPS) continue; // we don't include the null-symbol

        active_learning_namespace::update_tail(new_tail, symbol);
        new_tail->td->index = index;

        tail* old_tail = new_tail;
        new_tail = mem_store::create_tail(nullptr);
        new_tail->tr = new_trace;
        old_tail->set_future(new_tail);

        ++size;
    }

    new_tail->td->index = size;;
    new_trace->end_tail = new_tail;

    new_trace->length = size;

    new_trace->finalize();

    if(size==0){
        cerr << "Problematic prefix";
        print_vector(sequence);
        throw runtime_error("Size should always be larger 0 at this stage.");
    } 
}

/**
 * @brief What you think it does.
 * 
 */
vector<int> active_learning_namespace::concatenate_strings(const vector<int>& pref1, const vector<int>& pref2){
  vector<int> res(pref1);
  res.insert(res.end(), pref2.begin(), pref2.end());
  return res;
}

/**
 * @brief Overload. Does a positive (accepting) trace.
 * 
 * @param vec The vector.
 * @param id The inputdata.
 * @return trace* The trace.
 */
trace* active_learning_namespace::vector_to_trace(const vector<int>& vec, inputdata& id){
    return vector_to_trace(vec, id, knowledge_t::accepting);
}

/**
 * @brief Turns a vector to a trace.
 * 
 * @param vec The vector.
 * @param id The inputdata.
 * @param trace_type Accepting or rejecting.
 * @return trace* The trace.
 */
trace* active_learning_namespace::vector_to_trace(const vector<int>& vec, inputdata& id, const knowledge_t trace_type){
    static int trace_nr = 0;

    trace* new_trace = mem_store::create_trace(&id);
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
    new_trace->sequence = ++trace_nr;
    //++trace_nr;

    active_learning_namespace::add_sequence_to_trace(new_trace, vec);
    
    return new_trace;
}

/**
 * @brief Print a vector of ints. For debugging purposes.
 * 
 * @param v The vector.
 */
void active_learning_namespace::print_vector(const vector<int>& v){
    cout << "Here comes a vector: ";
    for(const auto symbol: v){
      cout << symbol << ",";
    }
    cout << endl;
}

/**
 * @brief For debugging purposes when using observation table. Prints columns of a row.
 * 
 * @param row A row of the observation table.
 */
void active_learning_namespace::print_all_columns(const std::map<pref_suf_t, knowledge_t>& row){
    cout << "Here come all columns in this row: ";
    for(const auto& col: row){
        print_vector(col.first);
    }
    cout << " ...end of columns of this row." << endl;
}