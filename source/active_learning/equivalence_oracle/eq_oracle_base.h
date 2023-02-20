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

#ifndef _SUL_BASE_H_
#define _SUL_BASE_H_

#include "sul_base.h"
#include "parameters.h"

#include <vector>
#include <optional>

class eq_oracle_base{
  protected:
    sul_base* sul;
    
    virtual void reset_sul() = 0; // TODO: change return type to what you need
  public:
    eq_oracle_base(sul_base* sul) : sul(sul){};
    virtual std::optional< std::vector<int> > find_counterexample(); // TODO: put in hypothesis

};

#endif
