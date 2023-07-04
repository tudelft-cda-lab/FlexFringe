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

#include <unordered_set>

/**
 * @brief Explanation in class desription, see header-file.
 * 
 * No loop in this version.
 */
/* refinement* evidence_based_strategy::perform(refinement_set* possible_refs, shared_ptr<node_to_refinement_map_T> node_to_ref_map) {
  refinement* top_ref = *(possible_refs->begin());
  apta_node* red_node = top_ref->red;

  // update the red and the blue state of the highest scoring refinement
  this->database_connector->update_state_with_statistics(red_node);

  if(dynamic_cast<merge_refinement*>(top_ref) != nullptr){
    apta_node* blue_node = dynamic_cast<merge_refinement*>(top_ref)->blue;
    this->database_connector->update_state_with_statistics(blue_node);
  }

  return this->merger->get_best_refinement();
} */

/**
 * @brief With a loop.
 */
refinement* evidence_based_strategy::perform(refinement_set* possible_refs, shared_ptr<node_to_refinement_map_T> node_to_ref_map) {
  refinement* top_ref = *(possible_refs->begin());
  apta_node* red_node = top_ref->red;
  
  apta_node* blue_node = nullptr;
  if(dynamic_cast<merge_refinement*>(top_ref) != nullptr){
    apta_node* blue_node = dynamic_cast<merge_refinement*>(top_ref)->blue;
  }

  unordered_set< refinement* >& refs_under_consideration = node_to_ref_map[red_node]; // it's ok to modify this list at this stage
  if(blue_node != nullptr){
    for(refinement* ref: node_to_ref_map[blue_node]) refs_under_consideration.insert(ref);
  }

  unordered_set<int> updated_nodes;
  for(auto ref: refs_under_consideration){
    apta_node* n = ref->red;
    if(!updated_nodes.contains(n->get_number())){
      this->database_connector->update_state_with_statistics(n);
      updated_nodes.insert(n->get_number());
    }

    if(dynamic_cast<merge_refinement*>(ref) != nullptr){
      n = dynamic_cast<merge_refinement*>(ref)->blue;
      if(!updated_nodes.contains(n->get_number())){
        this->database_connector->update_state_with_statistics(n);
        updated_nodes.insert(n->get_number());
      }
    } 
  }

  // TODO: finish this here

  // update the red and the blue state of the highest scoring refinement
/*   this->database_connector->update_state_with_statistics(red_node);

  if(dynamic_cast<merge_refinement*>(top_ref) != nullptr){
    apta_node* blue_node = dynamic_cast<merge_refinement*>(top_ref)->blue;
    this->database_connector->update_state_with_statistics(blue_node);
  } */

  return this->merger->get_best_refinement();
}