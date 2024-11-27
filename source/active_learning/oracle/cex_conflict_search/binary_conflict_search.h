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
    binary_conflict_search(const std::shared_ptr<conflict_detector_base>& cd) : conflict_search_base(cd) {
      throw std::logic_error("binary_conflict_search not implemented yet");
    };
};

#endif
