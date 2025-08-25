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

#ifdef __FLEXFRINGE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

//#include <map> // TODO: for debugging only
#include <unordered_map>
#include <memory>
#include <ranges>

/* The data contained in every node of the prefix tree or DFA */
class paul_data: public count_data {
  using layer_predictions_map = distinguishing_sequences_handler_base::layer_predictions_map;

  friend class paul_heuristic;

protected:
  REGISTER_DEC_DATATYPE(paul_data);
  float lm_confidence = -1; // confidence in prediction. -1 to check if initialized
  
  num_map inferred_final_counts;    
  int inferred_total_final;

  //std::vector<int> predictions;
#ifdef __FLEXFRINGE_CUDA
  using device_vector = distinguishing_sequences_handler_base::device_vector; // mapping to GPU memory instead
  device_vector predictions;
  ~paul_data() override;
#else
  layer_predictions_map predictions;
#endif 

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

  void add_inferred_type(const int t) noexcept;

  void add_tail(tail* t) override;

  void set_confidence(const float confidence) noexcept;
  float get_confidence() const noexcept { return lm_confidence; };

  inline bool has_type() const noexcept { return final_counts.size() > 0; }
  inline bool label_is_queried() const noexcept { return inferred_final_counts.size() > 0 && final_counts.size() == 0; }

  int predict_type(tail* t) override;

  void update(evaluation_data* right) override;
  void undo(evaluation_data* right) override;

  const auto& get_predictions() const noexcept {return predictions;}

  /**
   * @brief Gets the number of predictions made at length len.
   */
  const auto get_n_predictions(const int len) const noexcept {
  #ifdef __FLEXFRINGE_CUDA
    return this->predictions.len_size_map.at(len);  
  #else
    return this->predictions.at(len).size();
  #endif
  }

    /**
   * @brief Gets the total number of predictions.
   */
  const auto get_n_predictions() const noexcept {
    int res = 0;
  
#ifdef __FLEXFRINGE_CUDA
    for(const size_t size: this->predictions.len_size_map | std::ranges::views::values){
      res += static_cast<int>(size);
    }
#else
    for(const auto& preds: this->predictions | std::ranges::views::values){
      res += preds.size();
    }
#endif

  return res;
  }
  
  void set_predictions(layer_predictions_map&& predictions);
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
