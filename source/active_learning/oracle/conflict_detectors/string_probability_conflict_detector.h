/**
 * @file string_probability_conflict_detector.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Detects conflicts that arise in absolute difference of parsed string probability vs. the one returned from the SUL. 
 * Used e.g. in the publications "PDFA Distillation via String Probability Queries", Baumgartner and Verwer 2024, and 
 * "PDFA Distillation with Error Bound Guarantees", Baumgartner and Verwer 2024.
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_STRING_PROBABILITY_CONFLICT_DETECTOR_H_
#define _AL_STRING_PROBABILITY_CONFLICT_DETECTOR_H_

#include "conflict_detector_base.h" 

class string_probability_conflict_detector : public conflict_detector_base {
  private:
    inline const float get_string_prob(const std::vector<int>& substr, apta& hypothesis, inputdata& id) const;
  
  protected:
    std::pair<bool, std::optional<sul_response> > creates_conflict_common(const sul_response& resp, const std::vector<int>& substr, apta& hypothesis, inputdata& id) override;
  
  public:
    string_probability_conflict_detector(const std::shared_ptr<sul_base>& sul) : conflict_detector_base(sul) {};
    string_probability_conflict_detector(const std::shared_ptr<sul_base>& sul, const std::shared_ptr<ii_base>& ii_handler) 
    : conflict_detector_base(sul, ii_handler){};
};

#endif // _AL_STRING_PROBABILITY_CONFLICT_DETECTOR_H_