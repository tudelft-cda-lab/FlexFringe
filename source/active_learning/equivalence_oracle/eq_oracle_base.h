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

#include "sul_base.h"
#include "parameters.h"
#include "search_strategies/search_base.h"
#include "base_teacher.h"

#include "apta.h"
#include "state_merger.h"

#include <list>
#include <optional>
#include <utility>
#include <memory>

class eq_oracle_base{
  protected:
    std::shared_ptr<sul_base> sul;
    std::unique_ptr<search_base> search_strategy;

    virtual void reset_sul() = 0;
    virtual bool apta_accepts_trace(state_merger* merger, const list<int>& tr, inputdata& id) const = 0;
  public:
    eq_oracle_base(std::shared_ptr<sul_base>& sul) : sul(sul){};
    virtual std::optional< std::pair< std::list<int>, int> > equivalence_query(state_merger* merger, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher) = 0;
};

#endif
