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

  bool resp = false;
  if(AL_TEST_EMTPY_STRING) // IMPORTANT: The underlying oracle also needs to check this
    resp = conflict_detector->creates_conflict(current_substring, hypothesis, id).first;
  else if(!AL_TEST_EMTPY_STRING && cex.empty())
    throw invalid_argument("WARNING: al_test_empty_string set to false, but oracle tests empty string. Check your implementation.");
  
  while(!resp){ // works because cex has been determined to lead to conflict already
    update_current_substring(cex);
    resp = conflict_detector->creates_conflict(current_substring, hypothesis, id).first;
  }

  optional<sul_response> sul_resp = conflict_detector->creates_conflict(current_substring, hypothesis, id).second; 
  return make_pair(current_substring, sul_resp.value());
}

/**
 * @brief Easy to read.
 */
void linear_conflict_search::update_current_substring(const vector<int>& cex) noexcept {
  int idx = current_substring.size();
  current_substring.push_back(cex[idx]);
}