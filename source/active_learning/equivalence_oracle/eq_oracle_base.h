/**
 * @file run_active_learning.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base class for the equivalence oracle.
 * @version 0.1
 * @date 2023-02-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _EQ_ORACLE_BASE_H_
#define _EQ_ORACLE_BASE_H_

#include "base_teacher.h"
#include "parameters.h"
#include "cex_search_strategies/search_strategy_headers.h"
#include "cex_conflict_search/conflict_search_base.h"
#include "sul_headers.h"

#include "apta.h"
#include "state_merger.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

class eq_oracle_base {
  protected:
    std::shared_ptr<sul_base> sul;
    std::unique_ptr<search_base> search_strategy;
    std::unique_ptr<conflict_search_base> conflict_searcher;

    virtual void reset_sul() = 0;

  public:
    eq_oracle_base(std::shared_ptr<sul_base>& sul) : sul(sul){};

    /**
     * @brief Poses the equivalence query. Returns counterexample cex and true answer to cex if no equivalence proven.
     *
     * @param merger The state-merger.
     * @param teacher The teacher.
     * @return std::optional< std::pair< std::vector<int>, int> > Counterexample if not equivalent, else nullopt.
     * Counterexample is pair of trace and the answer to the counterexample as returned by the SUL.
     */
    virtual std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher) = 0;

    virtual void initialize(state_merger* merger){
      this->search_strategy->initialize(merger);
    }
};

#endif
