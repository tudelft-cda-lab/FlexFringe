/**
 * @file base_teacher.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This is a basic implementation of a teacher. It asks simple membership queries, and returns true for yes and 
 * no for 'not a member'.
 * @version 0.1
 * @date 2023-02-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _BASE_TEACHER_H_
#define _BASE_TEACHER_H_

#include "sul_base.h"
#include "definitions.h"

#include <vector>
#include <memory>

class base_teacher{
  protected:
    std::shared_ptr<sul_base> sul;
  public:
    const int ask_membership_query(const active_learning_namespace::pref_suf_t& query, inputdata& id);
    const int ask_membership_query(const active_learning_namespace::pref_suf_t& prefix, const active_learning_namespace::pref_suf_t& suffix, inputdata& id);
    
    const double get_string_probability(const active_learning_namespace::pref_suf_t& query, inputdata& id);

    base_teacher(std::shared_ptr<sul_base>& sul) : sul(sul) {};
};

#endif
