/**
 * @file linear_conflict_search.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "linear_conflict_search.h"

using namespace std;

/**
 * @brief Searches for a conflict within a DFA via linear search from beginning to end.
 * 
 * We search for the shortest string that actually causes the error to happen.
 * 
 * @param cex The counterexample.
 * @param hypothesis The merged hypothesis.
 * @param teacher The teacher.
 * @param id The inputdata wrapper.
 * @return std::vector<int> A vector leading to the conflict.
 */
pair< vector<int>, optional<response_wrapper> > dfa_conflict_search_namespace::linear_conflict_search::get_conflict_string(const vector<int>& cex, apta& hypothesis, 
                                                             const unique_ptr<base_teacher>& teacher, inputdata& id){
  vector<int> substring;
  for(auto s: cex){
    substring.push_back(s);
    
    int true_val = get_teacher_response(substring, teacher, id);
    int pred_value = parse_dfa(substring, hypothesis, id); // TODO: we can do this one faster too via memoization

    if(true_val != pred_value){
      response_wrapper rep(response_type::int_response);
      rep.set_int(true_val);
      return make_pair(substring, make_optional(rep));
    }
      
  }

  return make_pair(cex, nullopt);
}