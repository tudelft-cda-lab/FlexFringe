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

using namespace std;

pair<bool, optional<sul_response> > type_overlap_conflict_detector::creates_conflict(const vector<int>& substr, apta& hypothesis, inputdata& id) {
  throw logic_error("type_overlap_conflict_detector::creates_conflict not implemented");
  int true_val = sul->do_query(substr, id).GET_INT();
  int pred_value = parse_dfa_for_type(substr, hypothesis, id); // TODO: we can do this one faster too via memoization

  if(true_val != pred_value)
    return make_pair(true, make_optional(sul_response(true_val)));
  
  return make_pair(false, nullopt);
}

/**
 * @brief What you think it does.
 */
void type_overlap_conflict_detector::set_ii_handler(const std::shared_ptr<ii_base>& ii_handler) noexcept {
  this->ii_handler = ii_handler;
}
