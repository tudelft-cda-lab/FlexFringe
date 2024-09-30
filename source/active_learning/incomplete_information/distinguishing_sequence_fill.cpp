/**
 * @file distinguishing_sequence_fill.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "distinguishing_sequence_fill.h"
#include "paul_heuristic.h"

#include "inputdatalocator.h"
#include "common_functions.h"

#include <optional>

/**
 * @brief Takes the two nodes, walks through their subtrees, and stores all the suffixes for which the two subtree disagree. 
 * If a suffix in not in the set of distinguishing sequences at the moment, then it will be added 
 */
void distinguishing_sequence_fill::pre_compute(list<int>& suffix, unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right, const int depth){
  const static int max_search_depth = MAX_AL_SEARCH_DEPTH;
  if(max_search_depth > 0 && (left->get_depth() > max_search_depth || right->get_depth() > max_search_depth)) // making sure we don't bust the transformer
    return;
    
  else if(seen_nodes.contains(left))
    return;
  
  seen_nodes.insert(left);
  auto r_data = dynamic_cast<paul_data*>(right->get_data());
  auto l_data = dynamic_cast<paul_data*>(left->get_data());

  //cout << "Seen nodes size: " << seen_node.size() << endl;

  if(r_data->label_is_queried() && l_data->label_is_queried()){
    // TODO: do we want this?
    return;
  }
  
  // in these two following if-clauses the side effect happens (see description)
  if(!r_data->has_type()){
    complete_node(right, aut, teacher);
  }
  if(!l_data->has_type()){
    complete_node(left, aut, teacher);
  }

  if(l_data->predict_type(nullptr) != r_data->predict_type(nullptr)){
    ds_ptr->add_sequence(suffix);
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
        pre_compute(suffix, seen_nodes, aut, teacher, left_child, right_child, depth+1);
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
        pre_compute(suffix, seen_nodes, aut, teacher, left_child, right_child, depth+1);
        suffix.pop_back();
      } 
    }
  }
}

/**
 * @brief Collect all sequences that distinguish the two states.
 */
void distinguishing_sequence_fill::pre_compute(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right){
  list<int> suffix;
  unordered_set<apta_node*> seen_nodes;
  pre_compute(suffix, seen_nodes, aut, teacher, left, right, 0);
}

/**
 * @brief Concatenates prefix and suffix efficiently, returns a new vector with the result. 
 */
vector<int> distinguishing_sequence_fill::concat_prefsuf(const vector<int>& pref, const vector<int>& suff) const {
  vector<int> res(pref.size() + suff.size()); 
  res.insert(res.end(), pref.begin(), pref.end());
  res.insert(res.end(), suff.begin(), suff.end());
  
  return res;
}

void distinguishing_sequence_fill::add_data_to_tree(std::unique_ptr<apta>& aut, const vector<int>& seq, const string& answer, const float confidence){
  static inputdata& id = *(inputdata_locator::get());

  id.add_type(answer);
  int reverse_type = id.get_reverse_type(answer);

  trace* new_trace = active_learning_namespace::vector_to_trace(seq, id, reverse_type);
  id.add_trace_to_apta(new_trace, aut.get(), false);

  // the target should exist because we just added that state
  apta_node* node = aut->get_root();
  tail* iter = new_trace->head;
  while(!iter->is_final()){
    node = node->get_child(iter->get_symbol())->find();
    iter = iter->future();
  }

  paul_data* data = dynamic_cast<paul_data*>(node->get_data());
  data->set_confidence(confidence);
}

/**
 * @brief Take all the distinguishing sequences you currently have, add them to the two nodes, and ask the transformer to fill those two out.
 * Afterwards, reset the distinguishing sequences back to their original state.
 */
void distinguishing_sequence_fill::complement_nodes(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right) {
  const static int MIN_BATCH_SIZE = 128;
  const static int MAX_LEN = 30;

  cout << "Complementing. Size of distinguished sequences: " << ds_ptr->size() << endl;

  auto left_access_trace = left->get_access_trace();
  auto right_access_trace = right->get_access_trace();
  
  const active_learning_namespace::pref_suf_t left_prefix = left_access_trace->get_input_sequence(true, false);
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  
  vector< vector<int> > queries;
  optional< vector<int> > suffix = ds_ptr->next();

  while(suffix){
    auto full_sequence = concat_prefsuf(left_prefix, suffix.value());
    if(full_sequence.size() < MAX_LEN)
      queries.push_back(move(full_sequence));

    full_sequence = concat_prefsuf(right_prefix, suffix.value());
    if(full_sequence.size() < MAX_LEN)
      queries.push_back(move(full_sequence));

    if(queries.size() >= MIN_BATCH_SIZE){ // MIN_BATCH_SIZE might be violated by plus one, hence min
      vector< pair<string, float> > answers = teacher->ask_type_confidence_batch(queries, *(inputdata_locator::get()));

      for(int i=0; i < queries.size(); ++i){
        add_data_to_tree(aut, queries[i], answers[i].first, answers[i].second);
      }

      queries.clear();
    }

    suffix = ds_ptr->next();
  }

  if(queries.size() == 0)
    return;

  vector< pair<string, float> > answers = teacher->ask_type_confidence_batch(queries, *(inputdata_locator::get()));
  for(int i=0; i < queries.size(); ++i){
    add_data_to_tree(aut, queries[i], answers[i].first, answers[i].second);
  }
}