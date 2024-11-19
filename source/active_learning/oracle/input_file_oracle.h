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

#include "oracle_base.h"
#include "parameters.h"

#include <optional>
#include <utility>
#include <vector>

/**
 * @brief A naive oracle. It can only use the input_file_sul, which stores a set of input traces
 * in a data structure, then use those for both queries and equivalence.
 * 
 */
class input_file_oracle : public oracle_base {
  protected:
    void reset_sul(){
        // we won't need this guy here
    };

  public:
    input_file_oracle(std::shared_ptr<sul_base>& sul) : oracle_base(sul) {
      assert(dynamic_cast<input_file_sul*>(sul.get()) != nullptr, "input_file_oracle needs an input_file_sul");
    };

    std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger) override;
};

#endif
