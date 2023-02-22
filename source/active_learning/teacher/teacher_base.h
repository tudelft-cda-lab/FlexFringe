/**
 * @file teacher_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _TEACHER_BASE_H_
#define _TEACHER_BASE_H_

#include "trace.h"
#include "definitions.h"

#include "abbadingo_sul.h"

class teacher_base{
  protected:
    abbadingo_sul sul;
  public:
    virtual const bool ask_query(const active_learning_namespace::pref_suf_t& prefix, const active_learning_namespace::pref_suf_t& suffix) const = 0;
};

#endif
