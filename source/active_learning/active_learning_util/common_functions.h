/**
 * @file common_functions.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _COMMON_FUNCTIONS_H_
#define _COMMON_FUNCTIONS_H_

#include "evaluate.h"
#include "parameters.h"
#include "apta.h"
#include "tail.h"
#include "trace.h"
#include "refinement.h"
#include "definitions.h"

#include <vector>

namespace active_learning_namespace{

  apta_node* get_child_node(apta_node* n, tail* t);
  bool aut_accepts_trace(trace* tr, apta* aut); 

  void reset_apta(state_merger* merger, const std::vector<refinement*> refs);
  const std::vector<refinement*> minimize_apta(state_merger* merger);

  std::vector<int> concatenate_strings(const std::vector<int>& pref1, const std::vector<int>& pref2);

  
  trace* vector_to_trace(const std::vector<int>& vec, inputdata& id);
  trace* vector_to_trace(const std::vector<int>& vec, inputdata& id, const active_learning_namespace::knowledge_t trace_type);

  void add_sequence_to_trace(trace* new_trace, const std::vector<int> sequence);
  void update_tail(/*out*/ tail* t, const int symbol);

  void print_vector(const vector<int>& v);
}

#endif