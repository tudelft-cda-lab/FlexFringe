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
 * @param id The inputdata wrapper.
 * @return std::vector<int>, sul_response A vector leading to the conflict including the corresponding SUL response.
 */
pair< vector<int>, sul_response> linear_conflict_search::get_conflict_string(const vector<int>& cex, apta& hypothesis, inputdata& id) {
  current_substring.clear();

  pair<bool, optional<sul_response>> resp = conflict_detector->creates_conflict(current_substring, hypothesis, id);
  while(!resp.first){
    update_current_substring(cex);
  }

  return make_pair(current_substring, resp.second.value());
}

/**
 * @brief Easy to read.
 */
vector<int> linear_conflict_search::update_current_substring(const vector<int>& cex) noexcept {
  int idx = current_substring.size();
  current_substring.push_back(cex[idx]);
}