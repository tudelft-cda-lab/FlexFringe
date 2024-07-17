/**
 * @file linear_state_query_conflict_search.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_LINEAR_STATE_QUERY_CONFLICT_SEARCH_H_
#define _AL_LINEAR_STATE_QUERY_CONFLICT_SEARCH_H_

#include "linear_conflict_search.h"

// We use namespace here because we want to use the same class names for PDFA cases.
namespace dfa_conflict_search_namespace {
  
  class linear_state_query_conflict_search : public linear_conflict_search {
    protected:
      int get_teacher_response(const std::vector<int>& cex, const std::unique_ptr<base_teacher>& teacher, inputdata& id) const override;

    public:
      linear_state_query_conflict_search(){};
  };

};

#endif
