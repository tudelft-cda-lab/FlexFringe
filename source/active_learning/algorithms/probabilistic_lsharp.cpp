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

#include "probabilistic_lsharp.h"
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

#include "log_alergia.h"

#include <list>
#include <stdexcept>

using namespace std;
using namespace active_learning_namespace;


/**
 * @brief Take a node and complete it wrt to the alphabet.
 * 
 */
void probabilistic_lsharp_algorithm::extend_fringe(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const vector< trace* >& traces) const {
  for(trace* new_trace: traces){
    id.add_trace_to_apta(new_trace, merger->get_aut(), false); // for-loop needed for probabilistic version
  }
  for(auto& [symbol, p]: static_cast<log_alergia_data*>(n->get_data())->get_final_distribution()){
    apta_node* next_node = n->get_child(symbol);
    static_cast<log_alergia_data*>(next_node->get_data())->update_final_prob(p);
  }
}

/**
 * @brief Adds statistics to a node, returns the traces that were queried along the way.
 * 
 * @param merger 
 * @param n 
 * @param id 
 * @param alphabet 
 * @return trace* 
 */
optional< vector<trace*> > probabilistic_lsharp_algorithm::add_statistics(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const vector<int>& alphabet) const {
  static unordered_set<apta_node*> completed_nodes;
  if(completed_nodes.contains(n)) return nullopt;

  vector<trace*> res(alphabet.size());
  for(const int symbol: alphabet){
    if(n->get_child(symbol) == nullptr){
      auto access_trace = n->get_access_trace();

      pref_suf_t seq;
      if(n->get_number() != -1) seq = access_trace->get_input_sequence(true, true);
      else seq.resize(1);
      
      seq[seq.size()-1] = symbol;
      
      trace* new_trace = vector_to_trace(seq, id);
      id.add_trace(new_trace);
      res.push_back(new_trace);

      const double new_prob = get_probability_of_last_symbol(new_trace, id, teacher, merger->get_aut());
      static_cast<log_alergia_data*>(n->get_data())->insert_final_probability(symbol, new_prob);
    }
  }
  log_alergia::normalize_final_probs(n);
  completed_nodes.insert(n);

  return make_optional(res);
}


/**
 * @brief Update the statistics of this node, doubling the size of it. This will give us better results in the next 
 * query.
 * 
 * @param merger 
 * @param n 
 * @param id 
 * @param alphabet 
 */
void probabilistic_lsharp_algorithm::update_state(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const std::vector<int>& alphabet) const {
  complete_state(merger, n, id, alphabet);
}

void probabilistic_lsharp_algorithm::preprocess_apta(unique_ptr<apta>& the_apta){
  
}

void probabilistic_lsharp_algorithm::run(inputdata& id){
  int n_runs = 1;
  
  auto eval = unique_ptr<evaluation_function>(get_evaluation());
  eval->initialize_before_adding_traces();
  
  auto the_apta = unique_ptr<apta>(new apta());
  auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));

  const vector<int> alphabet = id.get_alphabet();
  cout << "Alphabet: ";
  active_learning_namespace::print_sequence< vector<int>::const_iterator >(alphabet.cbegin(), alphabet.cend());

  // init the root node, s.t. we have blue states to iterate over
  optional< vector<trace*> > queried_traces = add_statistics(merger, the_apta->get_root(), id, alphabet);
  extend_fringe(merger, the_apta->get_root(), id, queried_traces.value());
  
  { // initialize the root node's final probability
    pref_suf_t seq;
    trace* new_trace = vector_to_trace(seq, id);
    id.add_trace(new_trace);
    id.add_trace_to_apta(new_trace, the_apta.get());

    const double new_prob = get_probability_of_last_symbol(new_trace, id, teacher, merger->get_aut());
    static_cast<log_alergia_data*>(the_apta->get_root()->get_data())->update_final_prob(new_prob);
  }
  

  while(ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS){
    if( n_runs % 100 == 0 ) cout << "Iteration " << n_runs + 1 << endl;
    
    bool no_isolated_states = true; // avoid iterating over a changed data structure (apta)
    list<refinement*> performed_refinements;
    
    for(blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it){

      const auto blue_node = *b_it;
      if(blue_node->get_size() == 0) continue; // This should never happen: TODO: Delete line
      
      // this is a difference with Vandraager (they only complete red nodes), but we need it for statistical methods
      complete_state(merger, blue_node, id, alphabet); // TODO: do we need this one here really? I can't see it at the moment why
      queried_traces = add_statistics(merger, blue_node, id, alphabet);

      refinement_set possible_refs;
      for(red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it){
        const auto red_node = *r_it;
        
        refinement* ref = merger->test_merge(red_node, blue_node);
        if(ref != nullptr){
          possible_refs.insert(ref);
          break;
        } 
      }

      //if( possible_refs.size()>0 ){
      //  refinement* best_merge = extract_best_merge(&possible_refs); // implicit assumption: prefers merges over extends
      //  best_merge->doref(merger.get());
      //  performed_refinements.push_back(best_merge);
      //}
      if(possible_refs.size()==0){
        // giving the blue nodes child states before going red
        extend_fringe(merger, blue_node, id, queried_traces.value());

        no_isolated_states = false;
        refinement* ref = mem_store::create_extend_refinement(merger.get(), blue_node);
        ref->doref(merger.get());
        
        //performed_refinements.push_back(ref);
      }

      /* {
        static int model_nr = 0;
        if(model_nr % 10 == 0) print_current_automaton(merger.get(), "model.", to_string(model_nr) + ".new_fringe"); // printing the final model each time
        ++model_nr;
      } */
    }

    if(no_isolated_states){ // build hypothesis
      /* {
        static int model_nr = 0;
        print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".before_ref");
      } */
      preprocess_apta();
      minimize_apta(performed_refinements, merger.get());

      {
        static int model_nr = 0;
        print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_ref"); // printing the final model each time
      }

      while(true){
        /* While loop to check the type. type is < 0 if sul cannot properly respond to query, e.g. when the string 
        we ask cannot be parsed in automaton. We ignore those cases, as they lead to extra states in hypothesis. 
        This puts a burden on the equivalence oracle to make sure no query is asked twice, else we end 
        up in infinite loop.*/
        optional< pair< vector<int>, int > > query_result = oracle->equivalence_query(merger.get(), teacher);
        if(!query_result){
          cout << "Found consistent automaton => Print." << endl;
          print_current_automaton(merger.get(), OUTPUT_FILE, ".final"); // printing the final model each time
          return;
        }

        const int type = query_result.value().second;
        if(type < 0) continue;

        const vector<int>& cex = query_result.value().first;
        cout << "Counterexample of length " << cex.size() << " found: ";
        print_vector(cex);
        proc_counterex(teacher, id, the_apta, cex, merger, performed_refinements, alphabet);
        break;
      }
    }

    ++n_runs;
    if(ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS){
      cout << "Maximum of runs reached. Printing automaton." << endl;
      for(auto top_ref: performed_refinements){
        top_ref->doref(merger.get());
      }
      print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
      return;
    }
  }
}