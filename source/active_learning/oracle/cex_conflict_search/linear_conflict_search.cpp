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

template bool linear_conflict_search::get_creates_conflict(const vector<int>& cex, apta& hypothesis, inputdata& id);
template bool linear_conflict_search::get_creates_conflict(const vector< vector<int> >& cex, apta& hypothesis, inputdata& id);

template pair< vector<int>, sul_response> linear_conflict_search::get_conflict_string_common(const vector<int>& cex, apta& hypothesis, inputdata& id);
template pair< vector<int>, sul_response> linear_conflict_search::get_conflict_string_common(const vector< vector<int> >& cex, apta& hypothesis, inputdata& id);

/**
 * @brief To override our virtual function.
 * Actual implementation in get_conflict_string_common()
 */
pair< vector<int>, sul_response> linear_conflict_search::get_conflict_string(const vector<int>& cex, apta& hypothesis, inputdata& id){
  return get_conflict_string_common(cex, hypothesis, id);
}

/**
 * @brief To override our virtual function.
 * Actual implementation in get_conflict_string_common()
 */
pair< vector<int>, sul_response> linear_conflict_search::get_conflict_string(const vector< vector<int> >& cex, apta& hypothesis, inputdata& id){
  return get_conflict_string_common(cex, hypothesis, id);
}

/**
 * @brief Easy to read.
 */
void linear_conflict_search::update_current_substring(const vector<int>& cex) noexcept {
  int idx = current_substring.size();
  current_substring.push_back(cex[idx]);
}

/**
 * @brief Easy to read.
 */
void linear_conflict_search::update_current_substring(const vector< vector<int> >& cex) noexcept {
  int idx = current_substring_batch_format.at(0).size();
  current_substring_batch_format.at(0).push_back(cex.at(0)[idx]);
}