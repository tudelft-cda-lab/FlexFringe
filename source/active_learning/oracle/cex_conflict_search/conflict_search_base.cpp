/**
 * @file conflict_search_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "conflict_search_base.h"
#include "common_functions.h"

using namespace std;

/**
 * @brief Parses the DFA, returns the prediction of the DFA.
 * 
 * @param seq The sequence.
 * @param hypothesis The hypothesis.
 * @return int The result.
 */
int conflict_search_base::parse_dfa_for_type(const vector<int>& seq, apta& hypothesis, inputdata& id){
  trace* test_tr = active_learning_namespace::vector_to_trace(seq, id, 0); // type-argument irrelevant here
  
  apta_node* n = hypothesis.get_root();
  tail* t = test_tr->get_head();     
  for (int j = 0; j < t->get_length(); j++) {
      n = active_learning_namespace::get_child_node(n, t);
      if (n == nullptr) {
        std::cout << "Tree not parsable. Returning unspecified prediction." << std::endl;
          return -1;
      }
      t = t->future();
  }
  
  int pred_val = n->get_data()->predict_type(t);
  return pred_val;
}