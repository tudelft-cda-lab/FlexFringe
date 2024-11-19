/**
 * @file paul_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _PAUL_ORACLE_H_
#define _PAUL_ORACLE_H_

#include "active_sul_oracle.h"
#include "parameters.h"

#include "linear_conflict_search.h"

#include <optional>
#include <utility>

class paul_oracle : public active_sul_oracle {
  private:
    inline int get_sul_response(const std::vector< std::vector<int> >& query_string, inputdata& id) const;

  protected:
    void reset_sul() override {};

  public:
    paul_oracle(std::shared_ptr<sul_base>& sul) : active_sul_oracle(sul) {
        conflict_searcher = std::make_unique<type_linear_conflict_searcher>(sul);
        conflict_detector = std::make_unique<type_conflict_detector>(sul);
    };

    std::optional<std::pair<std::vector<int>, sul_response>>
    equivalence_query(state_merger* merger) override;
};

#endif
