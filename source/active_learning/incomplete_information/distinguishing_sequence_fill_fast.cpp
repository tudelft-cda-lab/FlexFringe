/**
 * @file distinguishing_sequence_fill_fast.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "distinguishing_sequence_fill_fast.h"
#include "paul_heuristic.h"
#include "parameters.h"

#include "inputdatalocator.h"
#include "common_functions.h"

#include <optional>

using namespace std;

/**
 * @brief Takes the two nodes, walks through their subtrees, and stores all the suffixes for which the two subtree disagree. 
 * If a suffix in not in the set of distinguishing sequences at the moment, then it will be added 
 */
void distinguishing_sequence_fill_fast::pre_compute(list<int>& suffix, unordered_set<apta_node*>& seen_nodes, unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth) {
  const static int max_search_depth = MAX_AL_SEARCH_DEPTH;
  if(max_search_depth > 0 && (left->get_depth() > max_search_depth || right->get_depth() > max_search_depth)) // making sure we don't bust the transformer
    return;
    
  else if(seen_nodes.contains(left))
    return;
  
  seen_nodes.insert(left);
  auto r_data = dynamic_cast<paul_data*>(right->get_data());
  auto l_data = dynamic_cast<paul_data*>(left->get_data());

  if(r_data->label_is_queried() && l_data->label_is_queried()){
    // TODO: do we want this?
    return;
  }
  
  // in these two following if-clauses the side effect happens (see description)
  if(!r_data->has_type()){
    complete_node(right, aut);
  }
  if(!l_data->has_type()){
    complete_node(left, aut);
  }

  if(l_data->predict_type(nullptr) != r_data->predict_type(nullptr)){
    if(ds_ptr->add_sequence(suffix))
      m_suffixes.emplace_back(suffix.begin(), suffix.end());
  }

  // first do the right side
  for(auto it = right->guards_start(); it != right->guards_end(); ++it){
    if(it->second->target == nullptr) continue; // no child with that guard

    int symbol = it->first;
    apta_guard* right_guard = it->second;
    apta_guard* left_guard = left->guard(symbol, right_guard);
      
    if(left_guard == nullptr || left_guard->target == nullptr){
      continue; // we don't care?
    } 
    else {
      apta_node* left_target = left_guard->target;
      apta_node* right_target = right_guard->target;
      
      apta_node* left_child = left_target->find();
      apta_node* right_child = right_target->find();
      
      if(left_child != right_child){
        suffix.push_back(symbol);
        pre_compute(suffix, seen_nodes, aut, left_child, right_child, depth+1);
        suffix.pop_back();
      }
    }
  }

  // left side
  for(auto it = left->guards_start(); it != left->guards_end(); ++it){
    apta_node* target = it->second->target;
    if(target == nullptr || target->source != left) // the source check enables a parsing through the tree -> no loops and symmetry to right nodes
      continue;

    int symbol = it->first;
    apta_guard* left_guard = it->second;
    apta_guard* right_guard = right->guard(symbol, left_guard);
      
    if(right_guard == nullptr || right_guard->target == nullptr){
      continue; // we don't care?
    } 
    else {
      apta_node* left_target = left_guard->target;
      apta_node* right_target = right_guard->target;
      
      apta_node* left_child = left_target->find();
      apta_node* right_child = right_target->find();
      
      if(left_child != right_child){
        suffix.push_back(symbol);
        pre_compute(suffix, seen_nodes, aut, left_child, right_child, depth+1);
        suffix.pop_back();
      } 
    }
  }
}


/**
 * @brief Prerequisite to check_consistency. We already compute the distribution for the red node, 
 * saving us recomputation of the same distribution over and over again.
 */
vector<int> distinguishing_sequence_fill_fast::predict_node_with_sul(apta& aut, apta_node* node) {
  static inputdata& id = *inputdata_locator::get(); 

  auto right_access_trace = node->get_access_trace();
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  
  vector< vector<int> > queries;
  unordered_set<int> no_pred_idxs;
  vector<int> res;

  for(const auto& suffix: m_suffixes){
    if(right_prefix.size() + suffix.size() > MAX_LEN){
      no_pred_idxs.insert(queries.size()+no_pred_idxs.size()); // invalid prediction
      continue;
    }

    queries.push_back(active_learning_namespace::concatenate_vectors(right_prefix, suffix));
    if(queries.size() >= MIN_BATCH_SIZE){ // if min-batch size % 2 != 0 will be larger
      const sul_response response = sul->do_query(queries, *(inputdata_locator::get()));
        
      int answers_idx = 0;
      const vector<int>& answers = response.GET_INT_VEC();
      for(int i=0; i<answers.size()+no_pred_idxs.size(); ++i){
        if(no_pred_idxs.contains(i)){
          res.push_back(-1);
          continue;
        }
        res.push_back(answers[answers_idx]);
        ++answers_idx;
      }
          
      queries.clear();
      no_pred_idxs.clear();
    }
  }

  if(queries.size() > 0){
    const sul_response response = sul->do_query(queries, *(inputdata_locator::get()));
    
    int answers_idx = 0;
    const vector<int>& answers = response.GET_INT_VEC();
    for(int i=0; i<answers.size()+no_pred_idxs.size(); ++i){
      if(no_pred_idxs.contains(i)){
        res.push_back(-1);
        continue;
      }
      res.push_back(answers[answers_idx]);
      ++answers_idx;
    }
  }

  return res;
}

/**
 * @brief Predicts the distribution emanating from node using the automaton using 
 * the DS. If automaton cannot be parsed with the strings the prediction is -1.
 */
vector<int> distinguishing_sequence_fill_fast::predict_node_with_automaton(apta& aut, apta_node* node){
  static inputdata& id = *inputdata_locator::get(); 

  auto right_access_trace = node->get_access_trace();
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  
  vector< vector<int> > queries;
  unordered_set<int> no_pred_idxs;
  vector<int> res;

  for(const auto& suffix: m_suffixes){
    if(right_prefix.size() + suffix.size() > MAX_LEN){
      no_pred_idxs.insert(queries.size()+no_pred_idxs.size()); // invalid prediction
      continue;
    }

    queries.push_back(active_learning_namespace::concatenate_vectors(right_prefix, suffix));
    if(queries.size() >= MIN_BATCH_SIZE){ // if min-batch size % 2 != 0 will be larger
      
      int answers_idx = 0;
      vector<int> answers;
      for(auto query : queries){
        const int answer = active_learning_namespace::predict_type_from_trace(active_learning_namespace::vector_to_trace(query, id), &aut, id);
        answers.push_back(answer);
      }

      for(int i=0; i<answers.size()+no_pred_idxs.size(); ++i){
        if(no_pred_idxs.contains(i)){
          res.push_back(-1);
          continue;
        }
        res.push_back(answers[answers_idx]);
        ++answers_idx;
      }
          
      queries.clear();
      no_pred_idxs.clear();
    }
  }

  if(queries.size() > 0){
    const sul_response response = sul->do_query(queries, *(inputdata_locator::get()));
    
    int answers_idx = 0;
    vector<int> answers;
    for(auto query : queries){
      const int answer = active_learning_namespace::predict_type_from_trace(active_learning_namespace::vector_to_trace(query, id), &aut, id);
      answers.push_back(answer);
    }

    for(int i=0; i<answers.size()+no_pred_idxs.size(); ++i){
      if(no_pred_idxs.contains(i)){
        res.push_back(-1);
        continue;
      }
      res.push_back(answers[answers_idx]);
      ++answers_idx;
    }
  }

  return res;
}
