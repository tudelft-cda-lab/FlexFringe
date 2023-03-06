/**
 * @file input_file_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-03-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _INPUT_FILE_ORACLE_H_
#define _INPUT_FILE_ORACLE_H_

#include "eq_oracle_base.h"
#include "parameters.h"
#include "input_file_sul.h"

#include <vector>
#include <optional>

class input_file_oracle : eq_oracle_base {
  protected:    
    input_file_sul* sul;
    virtual void reset_sul(){}; // TODO: change return type to what you need
    bool apta_accepts_trace(state_merger* merger, const vector<int>& tr) const;
  public:
    input_file_oracle(sul_base* sul) : sul(dynamic_cast< sul_base* >(sul)){};
    virtual std::optional< std::vector<int> > equivalence_query(state_merger* merger); // TODO: put in hypothesis

};

#endif
