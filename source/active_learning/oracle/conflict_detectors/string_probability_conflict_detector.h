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
    /**
     * Some SULs can output the direct string probability, whereas others output only next symbol probabilities.
     * This flag signals us whether we need to compute the probability from the next-symbol-probabilities.
     */
    const bool SUL_OUTPUTS_STRING_PROBABILITY = false;

    inline const float get_string_prob_from_hypothesis(const std::vector<int>& substr, apta& hypothesis, inputdata& id) const;
  
  protected:
    std::pair<bool, std::optional<sul_response> > creates_conflict_common(const sul_response& resp, const std::vector<int>& substr, apta& hypothesis, inputdata& id) override;

    /**
     * If we need to compute the probability from the next-symbol-probabilities, we do it in this function. Therefore serves as alternative to 
     * creates_conflict_common() for this case.
     */
    virtual std::pair<bool, std::optional<sul_response> > creates_conflict_compute_step_by_step(const std::vector<int>& substr, apta& hypothesis, inputdata& id, const bool sul_requires_batch_input);
  
  public:
    string_probability_conflict_detector(const std::shared_ptr<sul_base>& sul) : conflict_detector_base(sul) {};
    string_probability_conflict_detector(const std::shared_ptr<sul_base>& sul, const std::shared_ptr<distinguishing_sequences_handler_base>& ii_handler) 
    : conflict_detector_base(sul, ii_handler){};

    std::pair<bool, std::optional<sul_response> > creates_conflict(const std::vector<int>& substr, apta& hypothesis, inputdata& id) override;
    std::pair<bool, std::optional<sul_response> > creates_conflict(const std::vector< std::vector<int> >& substr, apta& hypothesis, inputdata& id) override;
};

#endif // _AL_STRING_PROBABILITY_CONFLICT_DETECTOR_H_