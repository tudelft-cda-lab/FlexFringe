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
void probabilistic_lsharp_algorithm::complete_state(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const vector<int>& alphabet) const {
  static unordered_set<apta_node*> completed_nodes;
  if(completed_nodes.contains(n)) return;

  for(const int symbol: alphabet){
    if(n->get_child(symbol) == nullptr){
      auto access_trace = n->get_access_trace();

      pref_suf_t seq;
      if(n->get_number() != -1) seq = access_trace->get_input_sequence(true, true);
      else seq.resize(1);
      
      seq[seq.size()-1] = symbol;
      
      trace* new_trace = vector_to_trace(seq, id);
      id.add_trace(new_trace);
      id.add_trace_to_apta(new_trace, merger->get_aut(), false); // for-loop needed for probabilistic version

      const double new_prob = get_probability_of_last_symbol(new_trace, id, teacher, merger->get_aut());
      static_cast<log_alergia_data*>(n->get_data())->insert_final_probability(symbol, new_prob);
    }
  }
  log_alergia::normalize_final_probs(n);
  completed_nodes.insert(n);
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

void probabilistic_lsharp_algorithm::postprocess(){
  // TODO: here we compute all the probabilities afterwards
}