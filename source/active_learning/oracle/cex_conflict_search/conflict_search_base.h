/**
 * @file conflict_search_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_CONFLICT_SEARCH_BASE_H_
#define _AL_CONFLICT_SEARCH_BASE_H_

#include "source/input/inputdata.h"
#include "sul_base.h"

#include <memory>
#include <vector>
#include <utility>
#include <optional>

class conflict_search_base {
  protected: 
    std::shared_ptr<sul_base> sul;

    std::pair<bool, sul_response > found_conflict_string() = 0;
    std::vector<int> get_next_substring(const std::vector<int>& substr);
    std::pair<bool, std::optional<sul_response> > creates_conflict(const std::vector<int>& substr, apta& hypothesis);

  public:
    conflict_search_base(const std::shared_ptr<sul_base>& sul) : sul(sul){};
    std::pair< std::vector<int>, sul_response> get_conflict_string(const std::vector<int>& cex, apta& hypothesis, inputdata& id);
};

#endif
