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

#include "inputdata.h"

#include <fstream>
#include <vector>
#include <stdexcept>

class teacher_base;

class sul_base{
  friend class base_teacher;
  friend class eq_oracle_base;

  protected:
    const bool parses_input_file; // input files mainly for SULs with database

    virtual void post() = 0;
    virtual void step() = 0;

    virtual void reset() = 0;

    virtual bool is_member(const std::vector<int>& query_trace) const = 0; /* {
      return false;
    } */

    virtual const int query_trace(const std::vector<int>& query_trace, inputdata& id) const = 0; /* {
      return -1;
    } */

    std::ifstream get_input_stream() const;
    
  public:
    sul_base(const bool parses_input_file) : parses_input_file(parses_input_file){}; // abstract anyway

    virtual void pre(inputdata& id){
      if(!parses_input_file){
        throw std::logic_error("This function should not be called with kind of SUL, or set parses_input_file flag to true.");
      }
    };

    const bool has_input_file() const noexcept {
      return parses_input_file;
    }
};

#endif
