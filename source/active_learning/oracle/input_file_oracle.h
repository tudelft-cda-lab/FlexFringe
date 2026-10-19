/**
 * @file input_file_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-03-06
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _INPUT_FILE_ORACLE_H_
#define _INPUT_FILE_ORACLE_H_

#include "base_oracle.h"
#include "parameters.h"
#include "input_file_sul.h"

#include <optional>
#include <utility>
#include <vector>

/**
 * @brief A naive oracle. It can only use the input_file_sul, which stores a set of input traces
 * in a data structure, then use those for both queries and equivalence.
 * 
 */
class input_file_oracle : public base_oracle {
  public:
    input_file_oracle(const std::shared_ptr<sul_base>& sul) : base_oracle(sul) {
      if(dynamic_cast<input_file_sul*>(sul.get()) == nullptr)
        throw std::logic_error("input_file_oracle needs an input_file_sul");
    };

    std::optional<std::pair<std::vector<int>, sul_response>>
    equivalence_query(state_merger* merger) override;
};

#endif
