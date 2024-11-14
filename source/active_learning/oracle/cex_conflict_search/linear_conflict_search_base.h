/**
 * @file linear_conflict_search.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_LINEAR_CONFLICT_SEARCH_H_
#define _AL_LINEAR_CONFLICT_SEARCH_H_

#include "dfa_conflict_search_base.h"

// We use namespace here because we want to use the same class names for PDFA cases.
namespace dfa_conflict_search_namespace {
  protected:
    std::vector<int> get_next_substring(const std::vector<int>& substr) override; // TODO: This subclass should only override this method
    std::pair<bool, std::optional<response_wrapper>> creates_conflict(const std::vector<int>& substr, apta& hypothesis) = 0;
    
  class linear_conflict_search : public dfa_conflict_search_base {
    public:
      linear_conflict_search(){};

      std::pair< std::vector<int>, std::optional<response_wrapper> > get_conflict_string(const std::vector<int>& cex, apta& hypothesis, inputdata& id) = 0;
  };
}

#endif
