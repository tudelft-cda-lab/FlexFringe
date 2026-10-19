/**
 * @file distinguishing_sequences_handler_fast.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "distinguishing_sequences_handler_fast.h"
#include "paul_heuristic.h"
#include "parameters.h"

#include "inputdatalocator.h"
#include "common_functions.h"

#include <optional>
#include <ranges>

#ifdef __FLEXFRINGE_CUDA
#include "distinguishing_sequences_gpu.cuh"
#endif

using namespace std;

/**
 * @brief Takes the two nodes, walks through their subtrees, and stores all the suffixes for which the two subtree disagree. 
 * If a suffix in not in the set of distinguishing sequences at the moment, then it will be added 
 */
void distinguishing_sequences_handler_fast::pre_compute(list<int>& suffix, unordered_set<apta_node*>& seen_nodes, unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth) {
  const static int max_search_depth = AL_MAX_SEARCH_DEPTH;
  if((LTD_LENGTH > 0 && suffix.size() == LTD_LENGTH) || (max_search_depth > 0 && (left->get_depth() > max_search_depth) || right->get_depth() > max_search_depth)) // making sure we don't bust the transformer
    return;
    
  else if(seen_nodes.contains(left))
    return;
  
  seen_nodes.insert(left);
  auto r_data = dynamic_cast<paul_data*>(right->get_data());
  auto l_data = dynamic_cast<paul_data*>(left->get_data());
  
  // in these two following if-clauses the side effect happens (see description)
  if(!r_data->has_type()){
    complete_node(right, aut);
  }
  if(!l_data->has_type()){
    complete_node(left, aut);
  }

  // TODO: the label_queried check below only makes sense when we have a faulty oracle
  if(l_data->predict_type(nullptr) != r_data->predict_type(nullptr) && !(r_data->label_is_queried() && l_data->label_is_queried())){
    m_suffixes.add_suffix(suffix);
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
vector<int> distinguishing_sequences_handler_fast::predict_node_with_sul(apta& aut, apta_node* node) {
  static inputdata& id = *inputdata_locator::get(); 

  auto right_access_trace = node->get_access_trace();
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  
  vector< vector<int> > queries;
  unordered_set<int> no_pred_idxs;
  vector<int> res;

  for(auto& [length, suffixes]: m_suffixes.get_suffixes()){
    for(const auto& suffix: suffixes){
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

  if(res.size()==0){ // this case will happen if MAX_LEN kills all possible queries
    res.assign(m_suffixes.size(), -1);
  }

  return res;
}

/**
 * @brief Prerequisite to check_consistency. We already compute the distribution for the red node, 
 * saving us recomputation of the same distribution over and over again.
 */
distinguishing_sequences_handler_fast::layer_predictions_map distinguishing_sequences_handler_fast::predict_node_with_sul_layer_wise(apta& aut, apta_node* node) {
  static inputdata& id = *inputdata_locator::get(); 

  auto right_access_trace = node->get_access_trace();
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  
  layer_predictions_map res;

  for(auto& [length, suffixes]: m_suffixes.get_suffixes()){
    vector< vector<int> > queries;
    unordered_set<int> no_pred_idxs;

    if(!res.contains(length))
      res[length] = vector<int>();

    auto& res_vec = res[length];

    for(const auto& suffix: suffixes){
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
            res_vec.push_back(-1);
            continue;
          }
          res_vec.push_back(answers[answers_idx]);
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
          res_vec.push_back(-1);
          continue;
        }
        res_vec.push_back(answers[answers_idx]);
        ++answers_idx;
      }
    }

    if(res_vec.size()==0){ // this case will happen if MAX_LEN kills all possible queries
      res_vec.assign(suffixes.size(), -1);
    }
  }

  return res;
}

/**
 * @brief Predicts the distribution emanating from node using the automaton using 
 * the DS. If automaton cannot be parsed with the strings the prediction is -1.
 */
vector<int> distinguishing_sequences_handler_fast::predict_node_with_automaton(apta& aut, apta_node* node){
  static inputdata& id = *inputdata_locator::get(); 

  auto right_access_trace = node->get_access_trace();
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  
  vector< vector<int> > queries;
  unordered_set<int> no_pred_idxs;
  vector<int> res;

  for(auto& [length, suffixes]: m_suffixes.get_suffixes()){
    for(const auto& suffix: suffixes){
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

  if(res.size()==0){ // this case will happen if MAX_LEN kills all possible queries
    res.assign(m_suffixes.size(), -1);
  }

  return res;
}

/**
 * @brief Predicts the distribution emanating from node using the automaton using 
 * the DS. If automaton cannot be parsed with the strings the prediction is -1.
 */
distinguishing_sequences_handler_fast::layer_predictions_map distinguishing_sequences_handler_fast::predict_node_with_automaton_layer_wise(apta& aut, apta_node* node){
  static inputdata& id = *inputdata_locator::get(); 

  auto right_access_trace = node->get_access_trace();
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  

  layer_predictions_map res;

  for(auto& [length, suffixes]: m_suffixes.get_suffixes()){
    vector< vector<int> > queries;
    unordered_set<int> no_pred_idxs;

    if(!res.contains(length))
      res[length] = vector<int>();

    auto& res_vec = res[length];

    for(const auto& suffix: suffixes){
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
            res_vec.push_back(-1);
            continue;
          }
          res_vec.push_back(answers[answers_idx]);
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
          res_vec.push_back(-1);
          continue;
        }
        res_vec.push_back(answers[answers_idx]);
        ++answers_idx;
      }
    }

    if(res_vec.size()==0){ // this case will happen if MAX_LEN kills all possible queries
      res_vec.assign(suffixes.size(), -1);
    }
  }

  return res;
}

/**
 * @brief Gets a threshold according to the policy.
 */
float distinguishing_sequences_handler_fast::compute_threshold(const optional<int>& d1, const optional<int>& d2){
  static const float initial_threshold = CHECK_PARAMETER;
  if(!AL_ADJUST_THRESHOLD || !d1 || !d2)
    return initial_threshold;

  auto max_depth = max(d1.value(), d2.value()) - 1;
  auto sigmoid_term = 1.0 / ( 1+exp(-0.3*(max_depth-5)) ) - 0.5;

  return initial_threshold + min(max(0.0, sigmoid_term), 0.5);
}


#ifdef __FLEXFRINGE_CUDA
/** 
 * Does the evaluation on GPU rather than CPU.
 */
bool distinguishing_sequences_handler_fast::distributions_consistent_layer_wise(const device_vector& d1, const device_vector& d2, 
                                                                                const optional<int> depth1_opt, const optional<int> depth2_opt) {
                                                                                  
  /* if(depth1_opt.has_value() && depth2_opt.has_value() && AL_ADJUST_THRESHOLD && (depth1_opt.value() >= 10 || depth2_opt.value() >= 10)){
    last_overlap = -1; // indicating we skipped this step
    return true;
  } */
  if(d1.len_size_map.size() != d2.len_size_map.size())
    throw runtime_error("Distributions do not match in lengths");

  float max_ratio = 0;
  for(auto len_size_t : d1.len_size_map | std::views::keys){
    const int len = static_cast<int>(len_size_t);
    const auto threshold = compute_threshold(depth1_opt, depth2_opt);

    if(!d2.len_size_map.contains(len))
      throw runtime_error("Distributions captured different lengths, this should not have happened");

    const auto n_preds = static_cast<int>(d1.len_size_map.at(len));
    if(n_preds != static_cast<int>(d1.len_size_map.at(len)))
      throw runtime_error("Distributions do not match in size in length " + to_string(len));

    const auto v1_d = d1.len_pred_map_d.at(len);
    const auto v2_d = d2.len_pred_map_d.at(len);

    const auto ratio = 1.0f - distinguishing_sequences_gpu::get_overlap_gpu(v1_d, v2_d, n_preds);
    if(ratio > threshold){
      //cout << "\nsize: " << v1.size() << ", depth: " << depth <<  ", ratio: " << ratio << endl;
      last_overlap = 0;
      return false;
    }
    
    max_ratio = max(max_ratio, ratio); // TODO: adjust the data types
  }

  last_overlap = 1-max_ratio;
  return true;
}
#else
/**
 * @brief Does what you think it does.
 * 
 * TODO: Describe how we do the check in particular.
 */
bool distinguishing_sequences_handler_fast::distributions_consistent_layer_wise(const layer_predictions_map& d1, const layer_predictions_map& d2, 
                                                                                const optional<int> depth1_opt, const optional<int> depth2_opt) {
                                                                                  
  /* if(depth1_opt.has_value() && depth2_opt.has_value() && AL_ADJUST_THRESHOLD && (depth1_opt.value() >= 10 || depth2_opt.value() >= 10)){
    last_overlap = -1; // indicating we skipped this step
    return true;
  } */
  if(d1.size() != d2.size())
    throw runtime_error("Distributions are unequal");

  float max_ratio = 0;
  for(auto len : d1 | std::views::keys){
    const auto threshold = compute_threshold(depth1_opt, depth2_opt);

    if(!d2.contains(len))
      throw runtime_error("Distributions captured different lengths, this should not have happened");

    const auto& v1 = d1.at(len);
    const auto& v2 = d2.at(len);
    if(v1.size() != v2.size())
      throw runtime_error("Distributions do not match in size in length " + to_string(len));

    const auto ratio = get_overlap(v1, v2);
    if(ratio > threshold){
      //cout << "\nsize: " << v1.size() << ", depth: " << depth <<  ", ratio: " << ratio << endl;
      last_overlap = 0;
      return false;
    }
    
    max_ratio = max(max_ratio, ratio); // TODO: adjust the data types
  }

  last_overlap = 1-max_ratio;
  //cout << "\nDisagreed: " << disagreed << " | agreed: " << agreed << "max ratio: " << max_ratio << endl;
  //cout << "\nmax ratio: " << max_ratio << endl;
  return true;
}
#endif