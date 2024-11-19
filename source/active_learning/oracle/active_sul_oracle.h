/**
 * @file active_sul_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This class is for SULs that are queriable. It utilizes a search strategy.
 * @version 0.1
 * @date 2023-04-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _ACTIVE_SUL_ORACLE_H_
#define _ACTIVE_SUL_ORACLE_H_

#include "oracle_base.h"
#include "parameters.h"

#include "linear_conflict_search.h"

#include <optional>
#include <utility>

class active_sul_oracle : public oracle_base {
  protected:
    void reset_sul() override {};

  public:
    active_sul_oracle(std::unique_ptr<sul_base>& sul) : oracle_base(sul) {
        conflict_searcher = std::make_unique<type_linear_conflict_searcher>(sul);
        conflict_detector = std::make_unique<type_conflict_detector>(sul);
    };

    std::optional<std::pair<std::vector<int>, sul_response>>
    equivalence_query(state_merger* merger) override;
};

#endif
