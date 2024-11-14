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

class input_file_oracle : public oracle_base {
  protected:
    virtual void reset_sul(){
        // we won't need this guy here
    };

  public:
    input_file_oracle(std::shared_ptr<sul_base>& sul) : oracle_base(sul) {
      //assert(dynamic_cast<input_file_sul*>(sul.get()) != nullptr);
      static_assert(std::is_same<input_file_sul, T.element_type>, "input_file_oracle needs an input_file_sul");
    };

    virtual std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger) override; // TODO: put in hypothesis
};

#endif
