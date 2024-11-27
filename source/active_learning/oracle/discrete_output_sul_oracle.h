/**
 * @file discrete_output_sul_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This class is for SULs that are queriable. It utilizes a search strategy.
 * @version 0.1
 * @date 2023-04-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _DISCRETE_OUTPUT_SUL_ORACLE_H_
#define _DISCRETE_OUTPUT_SUL_ORACLE_H_

#include "oracle_base.h"
#include "parameters.h"

#include "linear_conflict_search.h"
#include "type_conflict_detector.h"

#include <optional>
#include <utility>

class discrete_output_sul_oracle : public oracle_base {
  protected:
    void reset_sul() override {};

  public:
    discrete_output_sul_oracle(const std::shared_ptr<sul_base>& sul) : oracle_base(sul) {
        conflict_detector = std::make_shared<type_conflict_detector>(sul);
        conflict_searcher = std::make_unique<linear_conflict_search>(conflict_detector);
    };
};

#endif
