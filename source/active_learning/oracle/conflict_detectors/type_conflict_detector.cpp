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

pair<bool, optional<sul_reponse> > type_conflict_detector::creates_conflict(const vector<int>& substr, apta& hypothesis, inputdata& id) {
  int true_val = sul->do_query(substr, id).GET_INT();
  int pred_value = parse_dfa(substr, hypothesis, id); // TODO: we can do this one faster too via memoization

  if(true_val != pred_value)
    return make_pair(true, make_optional(sul_response(true_val)));
  
  return make_pair(false, nullopt);
}