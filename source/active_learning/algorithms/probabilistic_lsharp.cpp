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
  static unordered_set<apta_node*> extended_nodes; // TODO: we might not need this, since it is inherently backed in add_statistics and the size of traces
  if(extended_nodes.contains(n)) return;
  
  for(trace* new_trace: traces){
    id.add_trace_to_apta(new_trace, merger->get_aut(), false); // for-loop needed for probabilistic version
  }
  for(const auto& [symbol, p]: static_cast<log_alergia_data*>(n->get_data())->get_unmerged_distribution()){
    apta_node* next_node = n->get_child(symbol);
    static_cast<log_alergia_data*>(next_node->get_data())->update_final_prob(p);
  }
}

/**
 * @brief Adds statistics to a node, returns the traces that were queried along the way.
 * 
 * @param n The node to upate.
 * @return optional< vector<trace*> > A vector with all the new traces we added to the apta, or nullopt if we already processed this node. Saves runtime.
 */
optional< vector<trace*> > probabilistic_lsharp_algorithm::add_statistics(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const vector<int>& alphabet) const {
  static unordered_set<apta_node*> completed_nodes;
  if(completed_nodes.contains(n)) return nullopt;

  vector<trace*> res;
  for(const int symbol: alphabet){
    if(n->get_child(symbol) == nullptr){
      auto access_trace = n->get_access_trace();

      pref_suf_t seq;
      if(n->get_number() != -1 && n->get_number() != 0) seq = access_trace->get_input_sequence(true, true);
      else seq.resize(1);
      
      seq[seq.size()-1] = symbol;
      
      trace* new_trace = vector_to_trace(seq, id);
      id.add_trace(new_trace);
      res.push_back(new_trace);

      const double new_prob = get_probability_of_last_symbol(new_trace, id, teacher, merger->get_aut());      
      if(std::isnan(new_prob)) throw runtime_error("Error: NaN value has occurred."); // debugging
      
      static_cast<log_alergia_data*>(n->get_data())->insert_probability(symbol, new_prob);
    }
  }
  log_alergia::normalize_final_probs(static_cast<log_alergia_data*>( n->get_data() ));
  completed_nodes.insert(n);

  return make_optional(res);
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
void probabilistic_lsharp_algorithm::proc_counterex(const unique_ptr<base_teacher>& teacher, inputdata& id, unique_ptr<apta>& hypothesis, 
                                      const vector<int>& counterex, unique_ptr<state_merger>& merger, const refinement_list refs,
                                      const vector<int>& alphabet) const {
  // in this block we do a linear search for the fringe of the prefix tree. Once we found it, we ask membership queries for each substring
  // of the counterexample (for each new state that we create), and this way add the whole counterexample to the prefix tree
  reset_apta(merger.get(), refs);
  vector<int> substring;
  apta_node* n = hypothesis->get_root();
  for(auto s: counterex){
    if(n == nullptr){
      const auto queried_type = teacher->ask_membership_query(substring, id);
      trace* new_trace = vector_to_trace(substring, id, queried_type);
      id.add_trace_to_apta(new_trace, hypothesis.get(), false);
      id.add_trace(new_trace);
      substring.push_back(s);
    }
    else{
      // find the fringe
      substring.push_back(s);
      trace* parse_trace = vector_to_trace(substring, id, 0); // TODO: inefficient like this
      tail* t = substring.size() == 0 ? parse_trace->get_end() : parse_trace->get_end()->past_tail;
      n = active_learning_namespace::get_child_node(n, t);
      mem_store::delete_trace(parse_trace);
    }
  }

  // for the last element, too
  const auto queried_type = teacher->ask_membership_query(substring, id);
  trace* new_trace = vector_to_trace(substring, id, queried_type);
  id.add_trace_to_apta(new_trace, hypothesis.get(), false);
  id.add_trace(new_trace);

  // now let's walk over the apta again, completing all the states we created
  n = hypothesis->get_root();
  trace* parsing_trace = vector_to_trace(counterex, id);
  tail* t = parsing_trace->get_head();
  while(n != nullptr){
    optional< vector<trace*> > queried_traces = add_statistics(merger, n, id, alphabet);
    if(queried_traces) extend_fringe(merger, n, id, queried_traces.value());
    n = active_learning_namespace::get_child_node(n, t);
    t = t->future();
  }
}

/**
 * @brief Complete all the statistics of all the nodes that we have.
 * 
 * Works in two steps: In step 1, we iterate through all blue nodes and complete the 
 * statistics of those which are not complete, yet. 
 * In step 2, we perform a DFS through the unmerged automaton, pulling up the distributions from behind.
 * This way we get the best possible approximations of the outgoing probabilities per state.
 * 
 * @param the_apta The unmerged! apta.
 */
void probabilistic_lsharp_algorithm::preprocess_apta(unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta, inputdata& id, const vector<int>& alphabet){
  // step 1: complete statistics of blue nodes
  for(blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it){
    auto blue_node = *b_it;
    optional< vector<trace*> > queried_traces = add_statistics(merger, blue_node, id, alphabet);
  }

  // step 2: DFS and infer the outgoing probabilities per state
  stack<apta_node*> node_stack;
  unordered_set<apta_node*> seen_nodes;

  apta_node* n = the_apta->get_root();
  node_stack.push(n);
  seen_nodes.insert(n);
  while(!node_stack.empty()){
    n = node_stack.top();

    if(n->get_child(alphabet[0]) == nullptr){
      // no need to update the fringes probs
      node_stack.pop(); // this node is done
    }
    else if(seen_nodes.contains(n)){
      unordered_map<int, double> current_map;
      for(auto symbol: alphabet){
        const auto& future_probs = static_cast<log_alergia_data*>(n->get_child(symbol)->get_data())->get_outgoing_distribution();
        for(auto& [s, p]: future_probs) current_map[s] += p;
      }
      log_alergia::add_outgoing_probs(n, current_map);      
      node_stack.pop();
    }
    else{
      for(auto symbol: alphabet){
        apta_node* next_node = n->get_child(symbol);
        node_stack.push(next_node);
        seen_nodes.insert(next_node);
      }
    }
  }
}

/**
 * @brief Normalize all the probabilities so that they sum to 1 per state for each outgoing transition, and including the final probabilities.
 * 
 * @param the_apta The apta.
 */
void probabilistic_lsharp_algorithm::postprocess_apta(std::unique_ptr<apta>& the_apta){
  for(blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it){
    auto blue_node = *b_it;
    if(blue_node->get_number() == 11 || blue_node->get_number() == 12) cout << "Here we come!" << endl;
    log_alergia::normalize_outgoing_probs(static_cast<log_alergia_data*>( blue_node->get_data() ));
  }

  for(red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it){
    auto red_node = *r_it;
    log_alergia::normalize_outgoing_probs(static_cast<log_alergia_data*>( red_node->get_data() ));
  }
}

/**
 * @brief Resets the outgoing probabilities of the nodes to the original state that we obtain after
 * querying it in the add_statistics function.
 * 
 * Warnign: Needs to be called after resetting the apta.
 * 
 * @param hypothesis The merged apta.
 */
void probabilistic_lsharp_algorithm::reset_probabilities(unique_ptr<apta>& hypothesis) const {
  for(APTA_iterator it = APTA_iterator(hypothesis->get_root()); *it != nullptr; ++it){
    auto node = *it;
    static_cast<log_alergia_data*>( node->get_data() )->reset();
  }
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
  
  { // initialize the root node's final probability with empty string search
    pref_suf_t seq;
    trace* new_trace = vector_to_trace(seq, id);
    id.add_trace(new_trace);
    id.add_trace_to_apta(new_trace, the_apta.get());

    const double new_prob = teacher->get_string_probability(seq, id);
    static_cast<log_alergia_data*>(the_apta->get_root()->get_data())->update_final_prob(new_prob);
  }

  {
    static int model_nr = 0;
    print_current_automaton(merger.get(), "model.", "root");
  }

  preprocess_apta(merger, the_apta, id, alphabet);
  {
    static int model_nr = 0;
    print_current_automaton(merger.get(), "model.", "root_preprocessed");
  }

  while(ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS){
    if( n_runs % 100 == 0 ) cout << "Iteration " << n_runs + 1 << endl;
    
    bool no_isolated_states = true; // avoid iterating over a changed data structure (apta)
    list<refinement*> performed_refinements;
    
    for(blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it){

      const auto blue_node = *b_it;
      if(blue_node->get_size() == 0) continue; // This should never happen: TODO: Delete line
      
      //complete_state(merger, blue_node, id, alphabet); // TODO: do we need this one here really? I can't see it at the moment why
      queried_traces = add_statistics(merger, blue_node, id, alphabet);

      refinement_set possible_merges;
      for(red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it){
        const auto red_node = *r_it;
        
        refinement* ref = merger->test_merge(red_node, blue_node);
        if(ref != nullptr){
          possible_merges.insert(ref);
          break;
        } 
      }

      if(possible_merges.size()==0){
        // giving the blue nodes child states before going red
        if(queried_traces) extend_fringe(merger, blue_node, id, queried_traces.value());

        no_isolated_states = false;
        refinement* ref = mem_store::create_extend_refinement(merger.get(), blue_node);
        ref->doref(merger.get());
        
        //performed_refinements.push_back(ref); // red states stay
      }
    }

    if(no_isolated_states){ // build hypothesis
      {
        static int model_nr = 0;
        print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".before_pre");
      }

      cout << "Preprocessing apta" << endl;
      preprocess_apta(merger, the_apta, id, alphabet);

      {
        static int model_nr = 0;
        print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".before_ref");
      }

      cout << "Minimizing apta" << endl;
      minimize_apta(performed_refinements, merger.get());
      cout << "Postprocessing apta" << endl;
      postprocess_apta(the_apta);
      cout << "Testing hypothesis" << endl;

      {
        static int model_nr = 0;
        print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_ref"); // printing the final model each time
      }

      exit(1);

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
        reset_probabilities(the_apta);
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