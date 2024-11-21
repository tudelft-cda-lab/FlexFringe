/**
 * @file string_probability_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This class is for SULs that are queriable. It utilizes a search strategy.
 * @version 0.1
 * @date 2023-04-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _STRING_PROBABILITY_ORACLE_H_
#define _STRING_PROBABILITY_ORACLE_H_

#include "oracle_base.h"
#include "state_merger.h"
#include "parameters.h"

#include "string_probability_conflict_detector.h"
#include "linear_conflict_search.h"

#include <optional>
#include <utility>

class string_probability_oracle : public oracle_base {
  protected:
    virtual void reset_sul() override{};

  public:
    string_probability_oracle(const std::shared_ptr<sul_base>& sul) : oracle_base(sul) {
        conflict_detector = std::make_shared<string_probability_conflict_detector>(sul);
        conflict_searcher = std::make_unique<linear_conflict_search>(conflict_detector);
    };

    /* std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger); */
};

#endif
