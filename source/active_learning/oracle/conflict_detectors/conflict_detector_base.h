/**
 * @file conflict_class_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_CONFLICT_DETECTOR_BASE_H_
#define _AL_CONFLICT_DETECTOR_BASE_H_

#include "sul_base.h"

#include <memory>
#include <vector>
#include <utility>
#include <optional>

class conflict_detector_base {
  protected:
    std::shared_ptr<sul_base> sul;

  public:
    conflict_detector_base(const std::shared_ptr<sul_base>& sul) : sul(sul) {}; 
    virtual std::pair<bool, std::optional<sul_reponse> > creates_conflict(const std::vector<int>& substr, apta& hypothesis, inputdata& id) = 0;
};

#endif // _AL_CONFLICT_DETECTOR_BASE_H_