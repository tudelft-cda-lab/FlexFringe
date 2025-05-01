/**
 * @file type_conflict_detector.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "type_conflict_detector.h"
#include "common_functions.h"

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Walks throught the hypothesis with the substr, returns type of last node.
 */
const int type_conflict_detector::parse_dfa_for_type(const vector<int>& substr, apta& hypothesis, inputdata& id) {
  trace* tr = vector_to_trace(substr, id); // TODO: we could make this more efficient
  return predict_type_from_trace(tr, &hypothesis, id);
}

pair<bool, optional<sul_response> > type_conflict_detector::creates_conflict_common(const sul_response& resp, const vector<int>& substr, apta& hypothesis, inputdata& id) {
  const int true_val = resp.has_int_val() ? resp.GET_INT() : resp.GET_INT_VEC().at(0);
  const int pred_value = parse_dfa_for_type(substr, hypothesis, id); // TODO: we can do this one faster too via memoization

  if(true_val != pred_value)
    return make_pair(true, make_optional(sul_response(true_val)));
  
  return make_pair(false, nullopt);
}