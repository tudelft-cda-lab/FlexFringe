/**
 * @file paul_heuristic.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __PAUL_HEURISTIC__
#define __PAUL_HEURISTIC__

#include "count_types.h"

/* The data contained in every node of the prefix tree or DFA */
class paul_data: public count_data {

protected:
  REGISTER_DEC_DATATYPE(paul_data);
  float lm_confidence = -1; // confidence in prediction. -1 to check if initialized

public:
  void set_confidence(const float confidence) noexcept { lm_confidence = confidence; };
  float get_confidence() const noexcept { return lm_confidence; };

  inline bool has_type() const noexcept { return num_final() > 0; }
  inline bool label_is_queried() const noexcept { return lm_confidence > 0; }
};

class paul_heuristic : public count_driven {

protected:
  REGISTER_DEC_TYPE(paul_heuristic);
  
  
/* public:
  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);
  virtual void reset(state_merger *merger);
  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right, int depth); */
};

#endif
