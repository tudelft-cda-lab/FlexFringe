/**
 * @file refinement_selection_strategies.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "refinement_selection_strategies.h"
#include "state_merger.h"

/**
 * @brief Explanation in class desription, see header-file.
 */
refinement* evidence_based_strategy::perform(refinement_set* possible_refs, shared_ptr<node_to_refinement_map_T> node_to_ref_map) {
  refinement* top_ref = possible_refs->top();
  apta_node* red_node = top_ref->red;

  // update the red and the blue state of the highest scoring refinement
  this->database_connector->update_state_with_statistics(red);

  if(dynamic_cast<merge_refinement*>(top_ref) != nullptr){
    apta_node* blue_node = dynamic_cast<merge_refinement*>(top_ref)->blue;
    this->database_connector->update_state_with_statistics(blue);
  }

  state_merger* merger = this->aut->get_context();
  return merger->get_best_refinement();
}