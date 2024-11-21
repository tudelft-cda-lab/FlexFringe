/**
 * @file linear_conflict_search_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_LINEAR_CONFLICT_SEARCH_H_
#define _AL_LINEAR_CONFLICT_SEARCH_H_

#include "conflict_search_base.h"

class linear_conflict_search : public conflict_search_base {
  protected:
    std::vector<int> current_substring;
    inline std::vector<int> update_current_substring(const std::vector<int>& cex) noexcept;
    
  public:
    linear_conflict_search(const std::shared_ptr<conflict_detector_base>& cd) : conflict_search_base(cd) {};
      
    std::pair< std::vector<int>, sul_response> 
    get_conflict_string(const std::vector<int>& cex, apta& hypothesis, inputdata& id) override;
};

#endif
