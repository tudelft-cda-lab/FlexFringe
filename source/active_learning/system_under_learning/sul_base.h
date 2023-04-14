/**
 * @file sul_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base class for the system under learning.
 * @version 0.1
 * @date 2023-02-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _SUL_BASE_H_
#define _SUL_BASE_H_

#include "source/input/inputdata.h"

#include <fstream>
#include <list>
#include <stdexcept>

class teacher_base;

class sul_base{
  friend class base_teacher;
  friend class eq_oracle_base;

  protected:


    virtual void post() = 0;
    virtual void step() = 0;
    virtual void reset() = 0;

    virtual bool is_member(const std::list<int>& query_trace) const = 0;

    virtual const int query_trace(const std::list<int>& query_trace, inputdata& id) const = 0;

    std::ifstream get_input_stream() const;
    
  public:
    sul_base() = default; // abstract anyway

    virtual void pre(inputdata& id) = 0;
};

#endif
