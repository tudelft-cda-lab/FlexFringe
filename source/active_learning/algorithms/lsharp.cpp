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

const bool PRINT_ALL_MODELS = true;

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
      id.add_trace_to_apta(new_trace, merger->get_aut());
    }
  }
}

void lsharp_algorithm::proc_counterex(apta* aut, const std::list<int>& counterex) const {
  // TODO: this is a counterexample strategy. Do we need it here?
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
  
  auto eval = unique_ptr<evaluation_function>(get_evaluation()); // TODO: initialize
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
    if(n_runs % 100 == 0) cout << "Iteration " << n_runs + 1 << endl;
    
    bool no_isolated_states = true; // avoid iterating over a changed data structure (apta)
    for(blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it){

      const auto blue_node = *b_it;
      if(blue_node->get_size() == 0) continue;
      assert(!blue_node->is_red()); // TODO: delete. Responsibility of the iterator
      
      // this is a difference with Vandraager (they only complete red nodes), but we need it for statistical methods
      complete_state(merger, blue_node, id, alphabet);

      refinement_set possible_refs;
      for(red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it){
        const auto red_node = *r_it;
        assert(red_node->is_red()); // TODO: delete. Responsibility of the iterator

        complete_state(merger, red_node, id, alphabet);

        refinement* ref = merger->test_merge(red_node, blue_node);
        if(ref != nullptr) possible_refs.insert(ref);
      }

      if( possible_refs.size()>0 ){
        refinement* best_merge = extract_best_merge(&possible_refs);
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
        print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".not_final"); // printing the final model each time
      }

      // TODO: is the loop below guaranteed to terminate?
      while(true){
        /* While loop to check the type. type is < 0 if sul cannot properly respond to query, e.g. when the string 
        we ask cannot be parsed in automaton. We ignore those cases, as they lead to extra states in hypothesis. 
        This puts a burden on the equivalence oracle to make sure no quey is asked twice, else we end 
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
        auto cex_tr = vector_to_trace(cex, id, type);
        cout << "Found counterexample: " << cex_tr->to_string() << "\n"; //endl;

        reset_apta(merger.get(), refs); // note: does not reset the identified red states we had before
        id.add_trace_to_apta(cex_tr, merger->get_aut());
        break;
      }
    }

    ++n_runs;
    if(ENSEMBLE_RUNS > 0 && n_runs > ENSEMBLE_RUNS){
      cout << "Maximum of runs reached. Printing automaton." << endl;
      for(auto top_ref: refs){
        top_ref->doref(merger.get());
      }
      print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
    }
  }
}