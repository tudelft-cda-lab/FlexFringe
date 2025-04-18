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
#include "ds_initializer_factory.h"
#include "ds_initializer_registration.h"
#include "parameters.h"

#include "inputdatalocator.h"
#include "common_functions.h"

#include <optional>

using namespace std;

/**
 * @brief Takes the two nodes, walks through their subtrees, and stores all the suffixes for which the two subtree disagree. 
 * If a suffix in not in the set of distinguishing sequences at the moment, then it will be added 
 */
void distinguishing_sequence_fill::pre_compute(list<int>& suffix, unordered_set<apta_node*>& seen_nodes, unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth) {
  const static int max_search_depth = AL_MAX_SEARCH_DEPTH;
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
    //if(!ds_ptr->contains(suffix)) // TODO: We can use a bloom filter here for example...
    ds_ptr->add_suffix(suffix);
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
 * @brief Collect all sequences that distinguish the two states.
 */
void distinguishing_sequence_fill::pre_compute(unique_ptr<apta>& aut, apta_node* left, apta_node* right){
  if(!collect_suffixes())
    return;

  list<int> suffix; // need to pop from back element, therefore no forward list
  unordered_set<apta_node*> seen_nodes;
  pre_compute(suffix, seen_nodes, aut, left, right, 0);
}

/**
 * @brief Helper function used to add sequences to the tree (putting the data into the node's information).
 */
void distinguishing_sequence_fill::add_data_to_tree(unique_ptr<apta>& aut, const vector<int>& seq, const int reverse_type, const float confidence){
  static inputdata& id = *(inputdata_locator::get());

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
 * @brief Memoizes the suffixes. Saves us recomputation at the expense of memory and possible accuracy.
 * 
 */
/* void distinguishing_sequence_fill::memoize() noexcept {
  optional< vector<int> > suffix_opt = ds_ptr->next();
  while(suffix_opt){
    m_suffixes.push_back(move(suffix_opt.value()));
    suffix_opt = ds_ptr->next();
  }

  memoized = true;
} */

/**
 * @brief Gets a prediction of a node
 * 
 * @param aut 
 * @param node 
 * @return std::vector<int> 
 */
vector<int> distinguishing_sequence_fill::predict_node_with_sul(apta& aut, apta_node* node){
  static inputdata& id = *inputdata_locator::get(); 

  auto right_access_trace = node->get_access_trace();
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  
  vector< vector<int> > queries;
  unordered_set<int> no_pred_idxs;
  vector<int> res;

  optional< vector<int> > suffix = ds_ptr->next();
  while(suffix){
    if(right_prefix.size() + suffix.value().size() > MAX_LEN){
      suffix = ds_ptr->next();
      no_pred_idxs.insert(queries.size()+no_pred_idxs.size()); // invalid prediction
      continue;
    }

    queries.push_back(active_learning_namespace::concatenate_vectors(right_prefix, suffix.value()));
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

    suffix = ds_ptr->next();
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
vector<int> distinguishing_sequence_fill::predict_node_with_automaton(apta& aut, apta_node* node){
  static inputdata& id = *inputdata_locator::get(); 

  auto right_access_trace = node->get_access_trace();
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  
  vector< vector<int> > queries;
  unordered_set<int> no_pred_idxs;
  vector<int> res;

  optional< vector<int> > suffix = ds_ptr->next();
  while(suffix){
    if(right_prefix.size() + suffix.value().size() > MAX_LEN){
      suffix = ds_ptr->next();
      no_pred_idxs.insert(queries.size()+no_pred_idxs.size()); // invalid prediction
      continue;
    }

    queries.push_back(active_learning_namespace::concatenate_vectors(right_prefix, suffix.value()));
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

    suffix = ds_ptr->next();
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


/**
 * @brief Avoids duplicate code.
 */
bool distinguishing_sequence_fill::distributions_consistent(const std::vector<int>& v1, const std::vector<int>& v2) {
  if(v1.size() != v2.size())
    throw runtime_error("Something weird happened in predictions");
  
  int agreed = 0;
  int disagreed = 0;
  for(int i=0; i<v1.size(); ++i){
    if(v1[i]==-1 || v2[i]==-1){ // string could not be queried, perhaps too long for transformer?
      //cerr << "Invalid prediction likely resulting from non-parsable string. Deeming inconsistent." << endl;
      //return false;
      continue;
    }
    else if(v1[i] < -1 || v2[i] < -1)
      cerr << "Something weird happened in return value here. Please check." << endl;
      
    else if(v1[i] == v2[i])
      ++agreed;
    else
      ++disagreed;
  }

  float ratio = static_cast<float>(disagreed) / (static_cast<float>(disagreed) + static_cast<float>(agreed));

  static float threshold = CHECK_PARAMETER;
  //std::cout << "\n ratio: " << ratio << ", threshold: " << threshold << "size: " << v1.size() << std::endl;
  if(ratio > threshold){
    last_overlap = 0;
    cout << "Disagreed: " << disagreed << " | agreed: " << agreed << " | ratio: " << ratio << endl;
    return false;
  }
  
  last_overlap = 1-ratio;
  return true;
}

/**
 * @brief Prerequisite to check_consistency. We already compute the distribution for the red node, 
 * saving us recomputation of the same distribution over and over again.
 */
void distinguishing_sequence_fill::pre_compute(unique_ptr<apta>& aut, apta_node* node) {
  memoized_predictions = predict_node_with_sul(*aut, node);
}

/**
 * @brief Throw in all the distinguishing sequences, and see how many disagreements you do have on that.
 * @return true If consistent.
 * @return false If not consistent.
 */
bool distinguishing_sequence_fill::check_consistency(unique_ptr<apta>& aut, apta_node* left, apta_node* right){
  vector<int> predictions = predict_node_with_sul(*aut, left);
  return distributions_consistent(memoized_predictions, predictions);
}

/**
 * @brief Computes the overlap and returns it.
 */
double distinguishing_sequence_fill::get_score(){
  return last_overlap;
}

/**
 * @brief Collect a set of distinguishing sequences already, to make better decisions at the root level of 
 * the tree.
 */
void distinguishing_sequence_fill::initialize(std::unique_ptr<apta>& aut){
  auto initializer = ds_initializer_factory::get_initializer(AL_II_INITIALIZER_NAME);
  initializer->init(shared_from_this(), aut);
}

/**
 * @brief Adds the sequence given as the vector to the set of distinguishing sequences.
 * 
 * @param seq The sequence to add.
 */
void distinguishing_sequence_fill::add_suffix(const std::vector<int>& seq){
  ds_ptr->add_suffix(seq);
}

/**
 * @brief Under some circumstances, suffixes should not be collected anymore at some point.
 * 
 * @return true Collect suffixes still.
 * @return false Disable collecting suffixes.
 */
const bool distinguishing_sequence_fill::collect_suffixes() const {
  bool res = false;
  try{
    // if we use this initializer, we already collect a large set of sequences and rely on that
    res = AL_II_INITIALIZER_NAME != ds_initializer_registration::get_initializer_name(ds_initializer_registration::ds_initializers_t::pre_generate_sequences);
  }
  catch(...){
    throw invalid_argument("Do you have a valid initializer name for the ii_handler?");
  }

  return res;
} 



/**
 * @brief Take all the distinguishing sequences you currently have, add them to the two nodes, and ask the transformer to fill those two out.
 * Afterwards, reset the distinguishing sequences back to their original state.
 */
/*void distinguishing_sequence_fill::complement_nodes(unique_ptr<apta>& aut, apta_node* left, apta_node* right) {
  return; // we currently do not want this option

  auto left_access_trace = left->get_access_trace();
  auto right_access_trace = right->get_access_trace();
  
  const active_learning_namespace::pref_suf_t left_prefix = left_access_trace->get_input_sequence(true, false);
  const active_learning_namespace::pref_suf_t right_prefix = right_access_trace->get_input_sequence(true, false);
  
  vector< vector<int> > queries;
  optional< vector<int> > suffix = ds_ptr->next();

  while(suffix){
    auto full_sequence = active_learning_namespace::concatenate_vectors(left_prefix, suffix.value());
    if(full_sequence.size() < MAX_LEN)
      queries.push_back(move(full_sequence));

    full_sequence = active_learning_namespace::concatenate_vectors(right_prefix, suffix.value());
    if(full_sequence.size() < MAX_LEN)
      queries.push_back(move(full_sequence));

    if(queries.size() >= MIN_BATCH_SIZE){ // MIN_BATCH_SIZE might be violated by plus one, hence min
      const sul_response response = sul->do_query(queries, *(inputdata_locator::get()));
      const vector<int>& answers = response.GET_INT_VEC();
      const vector<double>& confidences = response.GET_DOUBLE_VEC();
      
      for(int i=0; i < queries.size(); ++i){
        add_data_to_tree(aut, queries[i], answers[i], confidences[i]);
      }

      queries.clear();
    }

    suffix = ds_ptr->next();
  }

  if(queries.size() == 0)
    return;

  const sul_response response = sul->do_query(queries, *(inputdata_locator::get()));
  const vector<int>& answers = response.GET_INT_VEC();
  const vector<double>& confidences = response.GET_DOUBLE_VEC();

  for(int i=0; i < queries.size(); ++i){
    add_data_to_tree(aut, queries[i], answers[i], confidences[i]);
  }
}*/
