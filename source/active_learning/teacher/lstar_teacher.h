/**
 * @file lstar_teacher.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _L_STAR_TEACHER_H_
#define _L_STAR_TEACHER_H_

#include "teacher_base.h"
#include "definitions.h"

#include <vector> 

class lstar_teacher : teacher_base {
  //protected:

  public:
    virtual const bool ask_query(const active_learning_namespace::pref_suf_t& prefix, const active_learning_namespace::pref_suf_t& suffix) const override;
};

#endif
