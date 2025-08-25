/**
 * @file type_overlap_conflict_detector.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "type_overlap_conflict_detector.h"
#include "common_functions.h"

using namespace std;
using namespace active_learning_namespace;

pair<bool, optional<sul_response> > type_overlap_conflict_detector::creates_conflict_common(const sul_response& resp, const vector<int>& substr, apta& hypothesis, inputdata& id) {
  apta_node* n = get_last_node(substr, &hypothesis, id);
  if(n==nullptr){
    cout << "Conflict because automaton not parsable" << endl;
    return make_pair(true, sul->do_query(substr, id));
  }
  
  const vector<int> d1 = ii_handler->predict_node_with_automaton(hypothesis, n);
  const vector<int>& d2 = resp.GET_INT_VEC();

  // TODO: we possibly tolerate mispredictions here. Dangerous game
  if(ii_handler->distributions_consistent(d1, d2))
    make_pair(false, nullopt);

  return make_pair(true, make_optional(sul->do_query(substr, id)));
}