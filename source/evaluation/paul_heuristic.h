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

#include <map> // TODO: for debugging only
#include <cmath> // TODO: for debugging only

/* The data contained in every node of the prefix tree or DFA */
class paul_data: public count_data {

  friend class paul_heuristic;

protected:
  REGISTER_DEC_DATATYPE(paul_data);
  float lm_confidence = -1; // confidence in prediction. -1 to check if initialized
  
  std::map<float, int> all_confidences; // TODO: for debugging only
  int num_merged = 0; // TODO: delete?

  num_map final_counts_backup;

  // TODO: delete function
  inline float map_confidence(const float c){
    int x = c * 100; // 0.94 becomes 94 e.g. 
    if(x % 10 > 5)
      return floor(float(x) / 10) / 10;
    else
      return ceil(float(x) / 10) / 10;
  }

public:
  void print_state_label(iostream& output) override;

  void set_confidence(const float confidence) noexcept { 
    lm_confidence = confidence;
    float q_c = map_confidence(confidence);
    all_confidences[confidence] = 1; 
  };

  void add_tail(tail* t) override;

  float get_confidence() const noexcept { return lm_confidence; };

  inline bool has_type() const noexcept { return num_final() > 0; }
  inline bool label_is_queried() const noexcept { return lm_confidence != -1; }

  void update(evaluation_data* right) override;
  void undo(evaluation_data* right) override;
};

class paul_heuristic : public count_driven {

protected:

  int check_for_consistency(paul_data* left, paul_data* right, int mismatch_count=0) const;

  REGISTER_DEC_TYPE(paul_heuristic);

  int n_inferred_inferred_pairs=0;
  int n_inferred_inferred_mismatches=0;
  
  int n_real_inferred_pairs=0;
  int n_real_inferred_mismatches=0;
  
  int n_real_real_pairs=0; // out of convenience

  bool overlap_consistent(apta_node* left, apta_node* right) const;
  
/* public:
  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);*/
public:
  bool consistent(state_merger* merger, apta_node* left, apta_node* right, int depth) override;
  double compute_score(state_merger* merger, apta_node* left, apta_node* right) override; 
  void reset(state_merger *merger) override;
};

#endif
