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

#include "discrete_output_sul_oracle.h"
#include "parameters.h"

#include "linear_conflict_search.h"
#include "type_overlap_conflict_detector.h"

#include <optional>
#include <utility>

class paul_oracle : public discrete_output_sul_oracle {
  private:
    //inline int get_sul_response(const std::vector< std::vector<int> >& query_string, inputdata& id) const;

  protected:
    void reset_sul() override {};

  public:
    paul_oracle(const std::shared_ptr<sul_base>& sul, const std::shared_ptr<ii_base>& ii_handler) : discrete_output_sul_oracle(sul) {
      if(!ii_handler)
        throw std::invalid_argument("ERROR: ii_handler not provided to paul oracle, but it depends on it.");

      conflict_detector = std::make_shared<type_overlap_conflict_detector>(sul, ii_handler); // TODO: get the ii_handler in here
      conflict_searcher = std::make_unique<linear_conflict_search>(conflict_detector);
    };

    //std::optional<std::pair<std::vector<int>, sul_response>>
    //equivalence_query(state_merger* merger) override;
};

#endif
