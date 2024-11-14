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
  protected:
    std::vector<int> get_next_substring(const std::vector<int>& substr) override; // TODO: This subclass should only override this method
    std::pair<bool, std::optional<response_wrapper> > creates_conflict(const std::vector<int>& substr, apta& hypothesis) = 0;

  public:
    binary_conflict_search(const std::shared_ptr<sul_base>& sul) : conflict_search_base(sul) {
      throw std::exception("binary_conflict_search not implemented yet");
    };
};

#endif
