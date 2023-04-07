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
 * @param aut 
 */
void lsharp_algorithm::complete_state(unique_ptr<state_merger>& merger, apta_node* n, base_teacher& teacher, inputdata& id, const vector<int>& alphabet) const {
  for(const int symbol: alphabet){
    if(n->get_child(symbol) == nullptr){
      auto access_trace = n->get_access_trace();

      pref_suf_t seq;
      if(n->get_number() != -1) seq = access_trace->get_input_sequence(true);

/*       if(n->get_number() != -1) cout << "node number: " << n->get_number() << ", access trace: " << access_trace->to_string() << ", seq: ";
      print_vector(seq); */

      seq.push_back(symbol);
      const auto answer = teacher.ask_membership_query(seq, id);
      
      trace* new_trace = vector_to_trace(seq, id, answer);
      //cout << "Trace to apta: " << new_trace->to_string() << endl;

      id.add_trace_to_apta(new_trace, merger->get_aut(), set<int>());
    }
  }
}

void lsharp_algorithm::proc_counterex(apta* aut, const std::vector<int>& counterex) const {
  // TODO: this is a counterexample strategy
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

void lsharp_algorithm::run(inputdata&& id){
  int n_runs = 0;

  // TODO: make those dynamic later
  base_teacher teacher(sul.get()); // TODO: make these generic when you can
  input_file_oracle oracle(sul.get()); // TODO: make these generic when you can

  if(sul->has_input_file()){
    sul->pre(id);
  }
  
  auto eval = unique_ptr<evaluation_function>(get_evaluation());
  auto the_apta = unique_ptr<apta>(new apta());
  auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));

  const vector<int> alphabet = id.get_alphabet();
  cout << "Alphabet: ";
  print_vector(alphabet);

  // init the root node, s.t. we have blue states to iterate over
  complete_state(merger, the_apta->get_root(), teacher, id, alphabet);
  while(ENSEMBLE_RUNS > 0 && n_runs < ENSEMBLE_RUNS){
    if(n_runs % 100 == 0) cout << "Iteration " << n_runs << endl;
    
    bool no_isolated_states = true; // avoid iterating over a changed data structure
    for(blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it){

      const auto blue_node = *b_it;
      if(blue_node->get_size() == 0) continue;
      assert(!blue_node->is_red()); 

      //if(!blue_node->is_blue()) continue; // blue_state_iterator gives us blue and white nodes
      //cout << "blue node: " << blue_node->get_number() << endl;
      
      // this is a difference with Vandraager, but we need it for statistical methods
      complete_state(merger, blue_node, teacher, id, alphabet);

      refinement_set possible_refs;
      for(red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it){
        const auto red_node = *r_it;
        assert(red_node->is_red());

        complete_state(merger, red_node, teacher, id, alphabet);

        refinement* ref = merger->test_merge(red_node, blue_node);
        if(ref != nullptr) possible_refs.insert(ref);
      }

      if( possible_refs.size()>0 ){
        refinement* best_merge = extract_best_merge(&possible_refs);
        if(dynamic_cast<extend_refinement*>(best_merge) != 0){
          best_merge->doref(merger.get());
          no_isolated_states = false;
        }
      }
    }

    if(no_isolated_states){
      // build hypothesis
      const list< refinement* > refs = minimize_apta(merger.get());

      if(PRINT_ALL_MODELS){
        static int model_nr = 0;
        print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".not_final"); // printing the final model each time
      }

      optional< pair< vector<int>, int > > query_result = oracle.equivalence_query(merger.get());
      if(!query_result){
        cout << "Found consistent automaton => Print." << endl;
        print_current_automaton(merger.get(), OUTPUT_FILE, ".final"); // printing the final model each time
        return;
      }
      else{
        const vector<int>& cex = query_result.value().first;
        const int type = query_result.value().second;
        auto cex_tr = vector_to_trace(cex, id, type);
        cout << "Found counterexample: " << cex_tr->to_string() << "\n"; //endl;

        reset_apta(merger.get(), refs); // note: does not reset the identified red states we had before

        if(PRINT_ALL_MODELS){
          static int model_nr = 0;
          print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_undo");
        }
        id.add_trace_to_apta(cex_tr, merger->get_aut(), set<int>());
      }
    }

    ++n_runs;
  }
  cout << "Maximum of runs reached. Printing automaton." << endl;
  print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
}