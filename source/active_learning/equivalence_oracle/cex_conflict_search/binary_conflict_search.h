/**
 * @file binary_conflict_search.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_BINARY_CONFLICT_SEARCH_H_
#define _AL_BINARY_CONFLICT_SEARCH_H_

#include "conflict_search_base.h"

class binary_conflict_search : public conflict_search_base {
  public:
    binary_conflict_search(){};

    std::pair< std::vector<int>, std::optional<response_wrapper> > get_conflict_string(const std::vector<int>& cex, apta& hypothesis, const std::unique_ptr<base_teacher>& teacher, inputdata& id) override;
};

#endif
