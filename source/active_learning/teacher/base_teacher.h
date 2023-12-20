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

#include "definitions.h"
#include "sul_base.h"

#include <memory>
#include <vector>

class base_teacher {
  protected:
    std::shared_ptr<sul_base> sul;

  public:
    /* Learning acceptors */
    const int ask_membership_query(const active_learning_namespace::pref_suf_t& query, inputdata& id);
    const int ask_membership_query(const active_learning_namespace::pref_suf_t& prefix,
                                   const active_learning_namespace::pref_suf_t& suffix, inputdata& id);

    /* For learning weighted automata or PDFA */
    const double get_string_probability(const active_learning_namespace::pref_suf_t& query, inputdata& id);
    // const float get_symbol_probability(const active_learning_namespace::pref_suf_t& access_seq, const int symbol,
    // inputdata& id);
    const std::vector<float> get_weigth_distribution(const active_learning_namespace::pref_suf_t& access_seq,
                                                     inputdata& id);

    base_teacher(std::shared_ptr<sul_base>& sul) : sul(sul){};
};

#endif
