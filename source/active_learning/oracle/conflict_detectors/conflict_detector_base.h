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
#include "memory/incomplete_information/ii_base.h"

#include <memory>
#include <vector>
#include <utility>
#include <optional>

class conflict_detector_base {
  protected:
    std::shared_ptr<sul_base> sul;
    virtual std::pair<bool, std::optional<sul_response> > creates_conflict_common(const sul_response& resp, const std::vector<int>& substr, apta& hypothesis, inputdata& id) = 0;

  public:
    conflict_detector_base(const std::shared_ptr<sul_base>& sul) : sul(sul) {};
    conflict_detector_base(const std::shared_ptr<sul_base>& sul, const std::shared_ptr<ii_base>& ii_handler) : sul(sul) {
      std::cerr << "Info: This SUL does not support an incomplete information handler. Are you certain you picked the correct one?" << std::endl;
    };
    
    // we have two kinds of instantiations for the ones below, because different SULs can expect either a batch or a single vector
    virtual std::pair<bool, std::optional<sul_response> > creates_conflict(const std::vector<int>& substr, apta& hypothesis, inputdata& id);
    virtual std::pair<bool, std::optional<sul_response> > creates_conflict(const std::vector< std::vector<int> >& substr, apta& hypothesis, inputdata& id);
};

#endif // _AL_CONFLICT_DETECTOR_BASE_H_