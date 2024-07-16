/**
 * @file dfa_conflict_search_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "dfa_conflict_search_base.h"

/**
 * @brief Parses the DFA, returns the prediction of the DFA.
 * 
 * @param seq The sequence.
 * @param hypothesis The hypothesis.
 * @return int The result.
 */
int dfa_conflict_search_base::parse_dfa(const vector<int>& seq, apta& hypothesis, inputdata& id){
  trace* test_tr = active_learning_namespace::vector_to_trace(seq, id, 0); // type-argument irrelevant here
  
  apta_node* n = hypothesis.get_root();
  tail* t = test_tr->get_head();     
  for (int j = 0; j < t->get_length(); j++) {
      n = active_learning_namespace::get_child_node(n, t);
      if (n == nullptr) {
          cout << "Tree not parsable. Returning unspecified prediction." << endl;
          return -1;
      }
      t = t->future();
  }
  
  int pred_val = n->get_data()->predict_type(t);
  return pred_val;
}