/**
 * @file type_overlap_conflict_detector.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#ifndef _AL_TYPE_OVERLAP_CONFLICT_DETECTOR_H_
#define _AL_TYPE_OVERLAP_CONFLICT_DETECTOR_H_

#include "type_conflict_detector.h"
#include "sul_base.h"
#include "ii_base.h"

/**
 * @brief A conflict detector that also work with an ii_handler.
 */
class type_overlap_conflict_detector : public type_conflict_detector {
  protected:
    std::shared_ptr<ii_base> ii_handler;

  public: 
    type_overlap_conflict_detector(const std::shared_ptr<sul_base>& sul, const std::shared_ptr<ii_base>& ii_handler) 
    : type_conflict_detector(sul), ii_handler(ii_handler) { }; 

    type_overlap_conflict_detector(const std::shared_ptr<sul_base>& sul) : type_conflict_detector(sul) {
      throw std::invalid_argument("Error: type_overlap_conflict_detector relies on ii_handler, but not provided.");
    }
    
    std::pair<bool, std::optional<sul_response> > creates_conflict(const std::vector<int>& substr, apta& hypothesis, inputdata& id) override;
};

#endif // _AL_CONFLICT_DETECTOR_BASE_H_