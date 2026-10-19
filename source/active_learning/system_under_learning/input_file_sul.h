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

#include "source/input/inputdata.h"
#include "source/input/parsers/i_parser.h"
#include "sul_base.h"

#include <map>
#include <vector>

class input_file_sul : public sul_base {
  private:
    std::map<std::vector<int>, int> all_traces;

  public:
    void reset(){};

    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;

    void pre(inputdata& id) override;
    const std::map<std::vector<int>, int>& get_all_traces() const { return all_traces; }
};

#endif