/**
 * @file probabilistic_lsharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-10-09
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
 * This is the only function that creates new nodes (apart from the root node).
 * 
 */
void probabilistic_lsharp_algorithm::extend_fringe(std::unique_ptr<state_merger>& merger, apta_node* n, unique_ptr<apta>& the_apta, inputdata& id, const vector< trace* >& traces) const {
  static unordered_set<apta_node*> extended_nodes; // TODO: we might not need this, since it is inherently backed in add_statistics and the size of traces
  if(extended_nodes.contains(n)) return;
  
  for(trace* new_trace: traces){
    id.add_trace_to_apta(new_trace, merger->get_aut(), false); // for-loop needed for probabilistic version
  }
}

void probabilistic_lsharp_algorithm::update_final_probability(apta_node* n, apta* the_apta) const {
  auto access_trace = n->get_access_trace();

  apta_node* current_node = the_apta->get_root();
  tail* t = access_trace->get_head();

  double product = 1;
  while(t != nullptr){
    auto s = t->get_symbol();
    auto* data = static_cast<log_alergia_data*>(current_node->get_data());
    product *= data->get_normalized_probability(s);

    current_node = current_node->get_child(s);
    t = t->future();
  }

  assert(current_node == n);
  static_cast<log_alergia_data*>(current_node->get_data())->update_final_prob(product);
}

void probabilistic_lsharp_algorithm::init_final_prob(apta_node* n, apta* the_apta, inputdata& id) const {
  pref_suf_t seq;

  [[likely]]
  if(n->get_number()!=-1 && n->get_number()!=0){
    auto at = n->get_access_trace();
    seq = at->get_input_sequence(true, false);
  }

  trace* new_trace = vector_to_trace(seq, id);
  //id.add_trace(new_trace);
  //id.add_trace_to_apta(new_trace, the_apta);

  const double new_prob = teacher->get_string_probability(seq, id);
  static_cast<log_alergia_data*>(n->get_data())->init_access_probability(new_prob);
  //static_cast<log_alergia_data*>(n->get_data())->update_final_prob(new_prob, true);
  //update_final_probability(n, the_apta);
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
  auto access_trace = n->get_access_trace();

  pref_suf_t seq;
  if(n->get_number() != -1 && n->get_number() != 0) seq = access_trace->get_input_sequence(true, true);
  else seq.resize(1);

  auto* data = static_cast<log_alergia_data*>(n->get_data());
  data->initialize_distributions(alphabet);
  for(const int symbol: alphabet){
      
    seq[seq.size()-1] = symbol;
      
    trace* new_trace = vector_to_trace(seq, id);
    id.add_trace(new_trace);
    res.push_back(new_trace);

    const double new_prob = teacher->get_string_probability(seq, id); //get_probability_of_last_symbol(new_trace, id, teacher, merger->get_aut());      
    if(std::isnan(new_prob)) throw runtime_error("Error: NaN value has occurred."); // debugging

    data->update_probability(symbol, new_prob);
  }

  init_final_prob(n, merger->get_aut(), id);
  update_tree_recursively(n, merger->get_aut(), alphabet); // must be called after init_final_prob()
  log_alergia::normalize_probabilities(static_cast<log_alergia_data*>( n->get_data() ));

  completed_nodes.insert(n);
  return make_optional(res);
}

/**
 * @brief Takes a node n, and updates all the probabilities that lead to it.
 * 
 * This function must only be called from within add_statistics() and 
 * must be called after init_final_prob() [make sure that the node n's
 * final probability has been initialized].
 * 
 * @param n The node.
 * @param the_apta The apta.
 */
void probabilistic_lsharp_algorithm::update_tree_recursively(apta_node* n, apta* the_apta, const vector<int>& alphabet) const {
  auto access_trace = n->get_access_trace();
  if(access_trace==nullptr) return; // The root node

  stack<apta_node*> nodes_to_update;
  apta_node* current_node = the_apta->get_root();

  tail* t = access_trace->get_head();
  
  while(current_node != n){ // walk all the way to n, getting the states up to there
    nodes_to_update.push(current_node);
    current_node = current_node->get_child(t->get_symbol());
    t = t->future();
  }

  // update the nodes behind n
  const auto& data_to_add = static_cast<log_alergia_data*>(n->get_data())->get_outgoing_distribution();
  while(!nodes_to_update.empty()){
    current_node = nodes_to_update.top();
    if(current_node == n) continue;

    auto current_node_data = static_cast<log_alergia_data*>(current_node->get_data());

    for(int s=0; s<data_to_add.size(); ++s){
      current_node_data->add_probability(s, data_to_add[s]);
    }

    log_alergia::normalize_probabilities(static_cast<log_alergia_data*>( current_node->get_data() ));
    nodes_to_update.pop();
  }

  // now we have to update the final probabilities
  current_node = the_apta->get_root();
  t = access_trace->get_head();
  double product = 1;
  while(t != nullptr/* current_node->get_child(t->get_symbol()) != n */){ // don't update the final prob of n
    auto s = t->get_symbol();
    auto* data = static_cast<log_alergia_data*>(current_node->get_data());
    product *= data->get_normalized_probability(s);

    current_node = current_node->get_child(s);
    static_cast<log_alergia_data*>(current_node->get_data())->update_final_prob(product);
    // TODO: normalize again? We have two constraints to satisfy, which makes that a bit harder
    t = t->future();
  }
}


void probabilistic_lsharp_algorithm::update_tree_dfs(apta* the_apta, const vector<int>& alphabet) const {
  apta_node* n = the_apta->get_root();
  log_alergia::normalize_probabilities(static_cast<log_alergia_data*>(n->get_data()));

  stack<apta_node*> node_stack;
  stack<double> p_stack;

  for(auto s: alphabet){
    node_stack.push(n->get_child(s));
    p_stack.push(static_cast<log_alergia_data*>(n->get_data())->get_normalized_probability(s));
  }

  while(!node_stack.empty()){
    n = node_stack.top();
    double p = p_stack.top();
    auto data = static_cast<log_alergia_data*>(n->get_data());
    data->update_final_prob(p);
    log_alergia::normalize_probabilities(data);

    for(auto s: alphabet){
      node_stack.push(n->get_child(s));
      double new_p = p * static_cast<log_alergia_data*>(n->get_data())->get_normalized_probability(s);
      p_stack.push(new_p);
    }

    node_stack.pop();
    p_stack.pop();
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
void probabilistic_lsharp_algorithm::proc_counterex(const unique_ptr<base_teacher>& teacher, inputdata& id, unique_ptr<apta>& hypothesis, 
                                      const vector<int>& counterex, unique_ptr<state_merger>& merger, const refinement_list refs,
                                      const vector<int>& alphabet) const {
  // linear search to find fringe, then append new states
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
  //double product = 1;

  while(!t->is_final()){
    optional< vector<trace*> > queried_traces = add_statistics(merger, n, id, alphabet);
    if(queried_traces) extend_fringe(merger, n, hypothesis, id, queried_traces.value());

    //auto* data = static_cast<log_alergia_data*>(n->get_data());
    //product *= data->get_weight(t->get_symbol());

    n = active_learning_namespace::get_child_node(n, t);
    t = t->future();

    //if(n != nullptr) static_cast<log_alergia_data*>(n->get_data())->update_final_prob(product);
  }
}


/**
 * @brief Main routine of this algorithm.
 * 
 * @param id Inputdata.
 */
void probabilistic_lsharp_algorithm::run(inputdata& id){
  int n_runs = 1;
  
  auto eval = unique_ptr<evaluation_function>(get_evaluation());
  eval->initialize_before_adding_traces();
  
  auto the_apta = unique_ptr<apta>(new apta());
  auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));

  const vector<int> alphabet = id.get_alphabet();
  cout << "Alphabet: ";
  active_learning_namespace::print_sequence< vector<int>::const_iterator >(alphabet.cbegin(), alphabet.cend());

  {
    // init the root node
    auto root_node = the_apta->get_root();
    optional< vector<trace*> > queried_traces = add_statistics(merger, root_node, id, alphabet);
    extend_fringe(merger, root_node, the_apta, id, queried_traces.value());
  }

  list<refinement*> performed_refinements;
  while(ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS){
    if( n_runs % 100 == 0 ) cout << "Iteration " << n_runs + 1 << endl;

    stack< vector<trace*> > traces_to_add;
    state_set blue_states; // not needed logically, but we need to avoid breaking the apta data structure
    for(blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it){
      const auto blue_node = *b_it;      
      optional< vector<trace*> > queried_traces = add_statistics(merger, blue_node, id, alphabet);
      
      if(queried_traces){ // can fail through states that were added due to counterexample processing
        traces_to_add.push(std::move(queried_traces.value()));
      }
       blue_states.insert(blue_node);
    }

    // go through each newly found fringe node, see if you can merge or extend
    bool identified_red_node = false; // avoid iterating over a changed data structure (apta)
    for(auto blue_node : blue_states){
      refinement_set possible_merges;
      for(red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it){
        const auto red_node = *r_it;
        
        refinement* ref = merger->test_merge(red_node, blue_node);
        if(ref != nullptr){
          possible_merges.insert(ref);
        } 
      }

      if(possible_merges.size()==0){
        identified_red_node = true;
        // giving the blue nodes child states before going red
        refinement* ref = mem_store::create_extend_refinement(merger.get(), blue_node);
        ref->doref(merger.get());
        performed_refinements.push_back(ref);
      }
      else{
        // get the best refinement from the heap
        refinement* best_merge = *(possible_merges.begin());
        for(auto it : possible_merges){
            if(it != best_merge) it->erase();
        }
        best_merge->doref(merger.get());
        performed_refinements.push_back(best_merge);
      }
    }

    if(identified_red_node){
      // extend the fringe and turn all nodes red
      while(!traces_to_add.empty()){
        auto& new_traces = traces_to_add.top();
        for(auto& tr: new_traces)
          id.add_trace_to_apta(tr, the_apta.get(), false);
        traces_to_add.pop();
      }

      continue;
    }

    /* {
      static int model_nr = 0;
      cout << "Model nr " << model_nr + 1 << endl;
      print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_ref");
    } */
    // only merges performed, hence we can test our hypothesis
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

    // output the automaton if termination criteria met
    ++n_runs;
    if(ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS){
      cout << "Maximum of runs reached. Printing automaton." << endl;
      for(auto top_ref: performed_refinements){
        top_ref->doref(merger.get());
      }
      print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
      return;
    }

    while(!traces_to_add.empty()){
      auto& new_traces = traces_to_add.top();
      for(auto& tr: new_traces)
        id.add_trace_to_apta(tr, the_apta.get(), false);
      traces_to_add.pop();
    }
    
    performed_refinements.clear();
  }
}