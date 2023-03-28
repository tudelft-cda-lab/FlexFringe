/**
 * @file input_file_sul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This is a standard SUL that answers normal membership queries. It does only contain positive traces,
 * and assumes all traces that are not part of it as negative. The SUL is just a file to read from.
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
#include "inputdata.h"

#include <vector>
#include <map>

// only those should be able to access the system
class teacher_base;
class eq_oracle_base;

class input_file_sul : public sul_base {

  // only the teacher and the oracle should be able to influence the sul
  friend class base_teacher;
  friend class eq_oracle_base;

  protected:

    std::map< std::vector<int>, int > all_traces;

    virtual void preprocessing();
    virtual void postprocessing();
    virtual void step();

    virtual void reset(){};

    bool is_member(const std::vector<int>& query_trace) const override;
    const int query_trace(const std::vector<int>& query_trace, inputdata& id) const override;
  public:
    input_file_sul();

    virtual void parse_input(inputdata& id);
    const std::map< std::vector<int>, int >& get_all_traces() const {
      return all_traces;
    }
};

#endif