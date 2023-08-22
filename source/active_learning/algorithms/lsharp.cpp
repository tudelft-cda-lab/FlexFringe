/**
 * @file lsharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The (strategic) L#-algorithm, as described by Vandraager et al. (2022): "A New Approach for Active Automata Learning Based on Apartness"
 * @version 0.1
 * @date 2023-03-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "lsharp.h"
#include "base_teacher.h"
#include "input_file_sul.h"
#include "common_functions.h"
#include "input_file_oracle.h"

#include "state_merger.h"
#include "parameters.h"
#include "inputdata.h"
#include "mem_store.h"
#include "greedy.h"
#include "main_helpers.h"

#include <list>

const bool PRINT_ALL_MODELS = false;

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Take a node and complete it wrt to the alphabet.
 * 
 * TODO: shall we check if we already have this node? If yes, then we don't need to do anything.
 * 
 * @param aut 
 */
void lsharp_algorithm::complete_state(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const list<int>& alphabet) const {
  for(const int symbol: alphabet){
    if(n->get_child(symbol) == nullptr){
      auto access_trace = n->get_access_trace();

      pref_suf_t seq;
      if(n->get_number() != -1) seq = access_trace->get_input_sequence(true);

      seq.push_back(symbol);
      const int answer = teacher->ask_membership_query(seq, id);
      if(answer == -1) continue;
      
      trace* new_trace = vector_to_trace(seq, id, answer);
      id.add_trace_to_apta(new_trace, merger->get_aut(), false);
    }
  }
}

/**
 * @brief Processing the counterexample recursively in the binary search strategy 
 * as described by the paper.
 * 
 * Operations done directly on the APTA.
 * 
 * @param aut The merged APTA.
 * @param counterex The counterexample.
 */
void lsharp_algorithm::proc_counterex(const unique_ptr<base_teacher>& teacher, inputdata& id, unique_ptr<apta>& hypothesis, 
                                      const list<int>& counterex, unique_ptr<state_merger>& merger, const refinement_list refs) const {
  // TODO: do the binary search to minimize the automaton that we get out of it
  static const bool linear_search = true; // if true, analyze the counterexamples linearly from beginning

  if(linear_search){
    // for test purposes
    reset_apta(merger.get(), refs);
    list<int> processed_counterexample;
    apta_node* n = hypothesis->get_root();
    for(auto s: counterex){
      processed_counterexample.push_back(s);
      trace* test_tr = vector_to_trace(processed_counterexample, id, 0); // TODO: inefficient like this
      tail* t = test_tr->get_end()->past_tail;

      n = active_learning_namespace::get_child_node(n, t);
      if(n == nullptr){
        const auto queried_type = teacher->ask_membership_query(processed_counterexample, id);
        test_tr->type = queried_type;
        id.add_trace_to_apta(test_tr, hypothesis.get(), false);
        return;
      }

      mem_store::delete_trace(test_tr);
    }

    return;
  }

  // TODO: implement the binary search here
  // I need to parse the prefix tree and the hypothesis the same time now
}

refinement* lsharp_algorithm::extract_best_merge(refinement_set* rs) const {
  refinement *r = nullptr;
  if (!rs->empty()) {
      r = *rs->begin();
      for(auto it : *rs){
          if(r != it) it->erase();
      }
  }

  return r;
}

void lsharp_algorithm::run(inputdata& id){
  int n_runs = 1;
  
  auto eval = unique_ptr<evaluation_function>(get_evaluation());
  eval->initialize_before_adding_traces();
  
  auto the_apta = unique_ptr<apta>(new apta());
  auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));

  const list<int> alphabet = id.get_alphabet();
  cout << "Alphabet: ";
  active_learning_namespace::print_sequence< list<int>::const_iterator >(alphabet.cbegin(), alphabet.cend());

  // init the root node, s.t. we have blue states to iterate over
  complete_state(merger, the_apta->get_root(), id, alphabet);
  list< refinement* > refs;
  while(ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS){
    if( n_runs % 100 == 0 ) cout << "Iteration " << n_runs + 1 << endl;
    
    bool no_isolated_states = true; // avoid iterating over a changed data structure (apta)
    for(blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it){

      const auto blue_node = *b_it;
      if(blue_node->get_size() == 0) continue;
      
      // this is a difference with Vandraager (they only complete red nodes), but we need it for statistical methods
      complete_state(merger, blue_node, id, alphabet);

      refinement_set possible_refs;
      for(red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it){
        const auto red_node = *r_it;
        
        refinement* ref = merger->test_merge(red_node, blue_node);
        if(ref != nullptr) possible_refs.insert(ref);
      }

      if( possible_refs.size()>0 ){
        refinement* best_merge = extract_best_merge(&possible_refs); // implicit assumption: prefers merges over extends
        // identify states. These will be kept throughout the algorithm
        if(dynamic_cast<extend_refinement*>(best_merge) != 0){
          best_merge->doref(merger.get());
          no_isolated_states = false;
        }
      }
    }

    if(no_isolated_states){
      // build hypothesis
      if(PRINT_ALL_MODELS){
        static int model_nr = 0;
        print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".before_ref");
      }

      refs = minimize_apta(merger.get());

      if(PRINT_ALL_MODELS){
        static int model_nr = 0;
        print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_ref"); // printing the final model each time
      }

      // TODO: is the loop below guaranteed to terminate?
      while(true){
        /* While loop to check the type. type is < 0 if sul cannot properly respond to query, e.g. when the string 
        we ask cannot be parsed in automaton. We ignore those cases, as they lead to extra states in hypothesis. 
        This puts a burden on the equivalence oracle to make sure no query is asked twice, else we end 
        up in infinite loop.*/
        optional< pair< list<int>, int > > query_result = oracle->equivalence_query(merger.get(), teacher);
        if(!query_result){
          cout << "Found consistent automaton => Print." << endl;
          print_current_automaton(merger.get(), OUTPUT_FILE, ".final"); // printing the final model each time
          return;
        }

        const int type = query_result.value().second;
        if(type < 0) continue;

        const list<int>& cex = query_result.value().first;
        cout << "Counterexample of length " << cex.size() << " found: ";
        print_list(cex);
        proc_counterex(teacher, id, the_apta, cex, merger, refs);
        break;
      }
    }

    ++n_runs;
    if(ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS){
      cout << "Maximum of runs reached. Printing automaton." << endl;
      for(auto top_ref: refs){
        top_ref->doref(merger.get());
      }
      print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
      return;
    }
  }
}