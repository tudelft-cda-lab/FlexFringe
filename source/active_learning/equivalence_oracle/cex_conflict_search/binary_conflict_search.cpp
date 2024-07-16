/**
 * @file binary_conflict_search.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "binary_conflict_search.h"

using namespace std;

/**
 * @brief Searches for a conflict within a DFA via binary search.
 * 
 * We search for the shortest string that actually causes the error to happen.
 * 
 * @param cex The counterexample.
 * @param hypothesis The merged hypothesis.
 * @param teacher The teacher.
 * @param id The inputdata wrapper.
 * @return std::vector<int> A vector leading to the conflict.
 */
pair< vector<int>, optional<response_wrapper> > binary_conflict_search::get_conflict_string(const vector<int>& cex, apta& hypothesis, 
                                                             const unique_ptr<base_teacher>& teacher, inputdata& id){
  vector<int> substring;
  for(auto s: cex){
    substring.push_back(s);
    int true_val = teacher->ask_membership_query(substring, id);

    throw runtime_error("Not implemented yet.");
  }
}