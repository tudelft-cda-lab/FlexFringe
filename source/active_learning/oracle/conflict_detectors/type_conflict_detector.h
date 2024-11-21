/**
 * @file type_conflict_detector.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#ifndef _AL_TYPE_CONFLICT_DETECTOR_H_
#define _AL_TYPE_CONFLICT_DETECTOR_H_

#include "conflict_detector_base.h" 

class type_conflict_detector: public conflict_detector_base {
  public: 
    type_conflict_detector(const std::shared_ptr<sul_base>& sul) : conflict_detector_base(sul) {}; 
    std::pair<bool, std::optional<sul_response> > creates_conflict(const std::vector<int>& substr, apta& hypothesis, inputdata& id) override;
};

#endif // _AL_CONFLICT_DETECTOR_BASE_H_