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
#include "source/active_learning/memory/distinguishing_sequences/distinguishing_sequences_handler_base.h"

#include <map> // TODO: for debugging only
#include <memory>

class distinguishing_sequences_handler_base;

/* The data contained in every node of the prefix tree or DFA */
class paul_data: public count_data {

  friend class paul_heuristic;

protected:
  REGISTER_DEC_DATATYPE(paul_data);
  float lm_confidence = -1; // confidence in prediction. -1 to check if initialized
  
  num_map inferred_final_counts;    
  int inferred_total_final;

  std::vector<int> predictions;

  // TODO: delete function
  inline float map_confidence(const float c){
    int x = c * 100; // 0.94 becomes 94 e.g. 
    if(x % 10 > 5)
      return floor(float(x) / 10) / 10;
    else
      return ceil(float(x) / 10) / 10;
  }

public:
  paul_data() : count_data::count_data() {
    final_counts.clear();
  }

  void initialize() override {
    count_data::initialize();
    inferred_final_counts.clear();
    inferred_total_final = 0;
  }

  void print_state_label(std::iostream& output) override;

  void set_confidence(const float confidence) noexcept;
  void add_inferred_type(const int t) noexcept;

  void add_tail(tail* t) override;

  float get_confidence() const noexcept { return lm_confidence; };

  inline bool has_type() const noexcept { return final_counts.size() > 0; }
  inline bool label_is_queried() const noexcept { return inferred_final_counts.size() > 0 && final_counts.size() == 0; }

  int predict_type(tail* t) override;

  void update(evaluation_data* right) override;
  void undo(evaluation_data* right) override;

  const auto& get_predictions() const noexcept {return predictions;}
  
  template<typename T>
  void set_predictions(T&& predictions){this->predictions = std::forward<T>(predictions);}
};

class paul_heuristic : public count_driven {

protected:
  std::shared_ptr<distinguishing_sequences_handler_base> ii_handler;

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
  void provide_ds_handler(std::shared_ptr<distinguishing_sequences_handler_base>& ii_handler);
};

#endif
