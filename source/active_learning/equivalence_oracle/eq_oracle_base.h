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

#include "apta.h"
#include "state_merger.h"

#include <vector>
#include <optional>
#include <utility>

class eq_oracle_base{
  protected:
    sul_base* sul;

    virtual void reset_sul() = 0; // TODO: change return type to what you need
    virtual bool apta_accepts_trace(state_merger* merger, const vector<int>& tr, inputdata& id) const = 0;
  public:
    eq_oracle_base(sul_base* sul) : sul(sul){};
    virtual std::optional< std::pair< std::vector<int>, int> > equivalence_query(state_merger* merger) = 0;
};

#endif
