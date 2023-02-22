/**
 * @file base_teacher.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "base_teacher.h"

using namespace std;
using namespace active_learning_namespace;

const knowledge_t base_teacher::ask_membership_query(const sul_base& sul, const pref_suf_t& prefix, const pref_suf_t& suffix) {
  std::vector query(prefix);
  query.insert(query.end(), suffix.begin(), suffix.end());

  if(sul.is_member(query)){
    return knowledge_t::accepting;
  }
  return knowledge_t::rejecting;
}