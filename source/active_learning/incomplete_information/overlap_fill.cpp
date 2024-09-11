/**
 * @file overlap_fill.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-08-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "overlap_fill.h"
#include "paul_heuristic.h"

#include "inputdatalocator.h"
#include "common_functions.h"

#include <iostream>


/**
 * @brief We give this one a sequence, and it asks for a reponse and adds it to the tree. Use by both query_node and add_child_node, 
 * therefore avoids duplicate code.
 * 
 * @param node: Saves computation time. If optional s_opt is provided the child of n with edge is updated, else n itself.
 * @param seq The sequence of the node to add data to.
 * @param s_opt: If provided the child of n with edge is updated, else if nullopt then n itself is updated.
 */
void overlap_fill::add_data_to_tree(std::unique_ptr<apta>& aut, active_learning_namespace::pref_suf_t& seq, unique_ptr<base_teacher>& teacher, apta_node* node, optional<int> s_opt){
  static inputdata& id = *(inputdata_locator::get());

  pair<int, float> res = teacher->ask_membership_confidence_query(seq, id);
  int type = res.first;
  trace* new_trace = active_learning_namespace::vector_to_trace(seq, id, type);
  id.add_trace_to_apta(new_trace, aut.get(), false);

  // the target should exist because we just added that state
  float confidence = res.second;
  paul_data* data;
  if(s_opt)
    data = dynamic_cast<paul_data*>(node->guard(s_opt.value() /*,  right_guard */)->target->get_data());
  else
    data = dynamic_cast<paul_data*>(node->get_data());

  data->set_confidence(confidence);
}


/**
 * @brief Queries the given node to give it the data it needs.
 * 
 * @param node The node to query.
 */
void overlap_fill::complete_node(apta_node* node, std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher){  
  trace* access_trace = node->get_access_trace();
  active_learning_namespace::pref_suf_t seq;
  seq = access_trace->get_input_sequence(true, false);

  add_data_to_tree(aut, seq, teacher, node);
}


/**
 * @brief Adds a child node to node with the edge labeled with symbol. The new node is automatically
 * filled with information from the teacher (to avoid duplicate queries).
 * 
 * @param node The node.
 * @param symbol The symbol.
 */
void overlap_fill::add_child_node(std::unique_ptr<apta>& aut, apta_node* node, std::unique_ptr<base_teacher>& teacher, const int symbol){
  auto access_trace = node->get_access_trace();
  active_learning_namespace::pref_suf_t seq;
  seq = access_trace->get_input_sequence(true, true);
  seq[seq.size() - 1] = symbol;
            
  add_data_to_tree(aut, seq, teacher, node, make_optional<int>(symbol));
}


/**
 * @brief Fills non-existent state pairs with active learning queries.
 * 
 * Invariant: left is red node, right is blue node.
 * 
 * Depth: We only want to walk until MAX_DEPTH starting from nodes to complete, therefore we give depth starting at nodes.
 * 
 */
void overlap_fill::complement_nodes(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right, const int depth){
  if(MAX_DEPTH > 0 && depth == MAX_DEPTH)
    return;
  
  auto r_data = dynamic_cast<paul_data*>(right->get_data());
  auto l_data = dynamic_cast<paul_data*>(left->get_data());

  if(r_data->label_is_queried() && l_data->label_is_queried()){
    return;
  }
  
  if(!r_data->has_type()){
    complete_node(right, aut, teacher);
  }
  if(!l_data->has_type()){
    complete_node(left, aut, teacher);
  }

  // first do the right side
  for(auto it = right->guards_start(); it != right->guards_end(); ++it){
    if(it->second->target == nullptr) continue; // no child with that guard
    int symbol = it->first;
    apta_guard* right_guard = it->second;
    apta_guard* left_guard = left->guard(symbol, right_guard);
      
    if(left_guard == nullptr || left_guard->target == nullptr){
       add_child_node(aut, left, teacher, symbol);
    } 
    else {
      apta_node* left_target = left_guard->target;
      apta_node* right_target = right_guard->target;
      
      apta_node* left_child = left_target->find();
      apta_node* right_child = right_target->find();
      
      if(left_child != right_child){
        complement_nodes(aut, teacher, left_child, right_child, depth+1);
      } 
    }
  }

  // left side
  for(auto it = left->guards_start(); it != left->guards_end(); ++it){
    if(it->second->target == nullptr) continue; // no child with that guard
    int symbol = it->first;
    apta_guard* left_guard = it->second;
    apta_guard* right_guard = right->guard(symbol, right_guard);
      
    if(right_guard == nullptr || right_guard->target == nullptr){
       add_child_node(aut, right, teacher, symbol);
    } 
    else {
      apta_node* left_target = left_guard->target;
      apta_node* right_target = right_guard->target;
      
      apta_node* left_child = left_target->find();
      apta_node* right_child = right_target->find();
      
      if(left_child != right_child){
        complement_nodes(aut, teacher, left_child, right_child, depth+1);
      } 
    }
  }
}