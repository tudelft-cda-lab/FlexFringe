/**
 * @file teacher_imat.h
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief Teacher in the iMAT framework. The incomplete teacher. In this framework answering "don't know" (-1) is also
 * possible.
 * @version 0.1
 * @date 2024-04-12
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _TEACHER_IMAT_H_
#define _TEACHER_IMAT_H_
#include "base_teacher.h"
#include "definitions.h"
#include "sul_base.h"

class teacher_imat : public base_teacher {
  public:
    const int ask_membership_query(const active_learning_namespace::pref_suf_t& query, inputdata& id) override;
    teacher_imat(std::shared_ptr<sul_base>& sul) : base_teacher(sul){};
};

#endif
