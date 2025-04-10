/**
 * @file distinguishing_sequences.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "distinguishing_sequences.h"

using namespace std;

/**
 * @brief Making add_suffix shared code at least.
 * We need this so we can continue having add_suffix virtual, while also suppporting
 * templated/shared code.
 */
template<typename T> requires (is_same_v<T, list<int>> || is_same_v<T, vector<int>>)
bool add_suffix_impl(const T& s, suffix_tree& seq_store){
  if(s.size() == 0) return false;
  bool new_suffix_found = seq_store.add_suffix(s);
  return new_suffix_found;
}

/**
 * @brief Does what you think it does.
 * 
 * Returns true if a new suffix has been added, else false.
 */
bool distinguishing_sequences::add_suffix(const list<int>& s) noexcept {
  return add_suffix_impl(s, seq_store);
}

/**
 * @brief Does what you think it does.
 * 
 * Returns true if a new suffix has been added, else false.
 */
bool distinguishing_sequences::add_suffix(const vector<int>& s) noexcept {
  return add_suffix_impl(s, seq_store);
}