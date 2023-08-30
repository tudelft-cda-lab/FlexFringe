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

#include "sul_headers.h"
#include "search_strategies/search_strategy_headers.h"
#include "base_teacher.h"
#include "parameters.h"

#include "apta.h"
#include "state_merger.h"

#include <vector>
#include <optional>
#include <utility>
#include <memory>

class eq_oracle_base{
  protected:
    std::shared_ptr<sul_base> sul;
    std::unique_ptr<search_base> search_strategy;

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
    virtual std::optional< std::pair< std::vector<int>, int> > equivalence_query(state_merger* merger, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher) = 0;
};

#endif
