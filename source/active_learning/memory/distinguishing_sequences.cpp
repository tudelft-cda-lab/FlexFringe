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

/**
 * @brief Does what you think it does.
 * 
 * Returns true if a new suffix has been added, else false.
 */
bool distinguishing_sequences::add_sequence(const std::list<int>& s) noexcept {
  if(s.size() == 0) return false;
  bool new_suffix_found = seq_store.add_suffix(s);
  return new_suffix_found;
}