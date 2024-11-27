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
#include "conflict_detectors/conflict_detector_base.h" // why won't it find the path?

#include <memory>
#include <vector>
#include <utility>

class conflict_search_base {
  protected:
    std::shared_ptr<conflict_detector_base> conflict_detector;
    int parse_dfa_for_type(const std::vector<int>& seq, apta& hypothesis, inputdata& id);

  public:
    conflict_search_base(const std::shared_ptr<conflict_detector_base>& cd) : conflict_detector(cd) {};
    virtual std::pair< std::vector<int>, sul_response> get_conflict_string(const std::vector<int>& cex, apta& hypothesis, inputdata& id) = 0;
};

#endif // _AL_CONFLICT_SEARCH_BASE_H_