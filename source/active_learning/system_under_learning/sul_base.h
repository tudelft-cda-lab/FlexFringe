/**
 * @file run_active_learning.h
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

#include <fstream>
#include <vector>

class teacher_base;

class sul_base{
  friend class teacher_base;
  friend class base_teacher;

  protected:
    const bool parses_input_file = false; // input files mainly for SULs with database

    virtual void preprocessing(){};
    virtual void postprocessing(){};
    virtual void step(){};

    virtual void reset(){};

    virtual bool is_member(const std::vector<int>& query_trace) const {
      return true;
    }

    virtual bool has_input_file() const noexcept {
      return parses_input_file;
    }

    std::ifstream get_input_stream() const;
    
  public:
    sul_base() = delete;
};

#endif
