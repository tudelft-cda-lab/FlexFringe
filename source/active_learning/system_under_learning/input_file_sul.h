/**
 * @file positive_traces_sul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This is a standard SUL that answers normal membership queries. It does only contain positive traces,
 * and assumes all traces that are not part of it as negative.
 * @version 0.1
 * @date 2023-02-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _INPUT_FILE_SUL_H_
#define _INPUT_FILE_SUL_H_

#include "sul_base.h"
#include "i_parser.h"

#include <vector>
#include <set>

// only those should be able to access the system
class teacher_base;
class eq_oracle_base;

class positive_traces_sul : sul_base {
  friend class teacher_base;
  friend class eq_oracle_base;

  protected:
    std::set< std::vector<int> > all_traces;

    virtual void preprocessing();
    virtual void postprocessing();
    virtual void step();

    void parse_input();
    bool is_member(const std::vector<int>& query_trace) const;
  public:
    positive_traces_sul();
};

#endif