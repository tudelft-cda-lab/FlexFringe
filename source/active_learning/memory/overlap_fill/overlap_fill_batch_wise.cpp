/**
 * @file overlap_fill_batch_wise_batch_wise.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "overlap_fill_batch_wise.h"
#include "paul_heuristic.h"

#include "inputdatalocator.h"
#include "common_functions.h"
#include "parameters.h"

#include <iostream>

using namespace std;

/**
 * @brief We give this one a sequence, and it asks for a reponse and adds it to the tree. Use by both query_node and add_child_node, 
 * therefore avoids duplicate code.
 * 
 * @param node: Saves computation time. If optional s_opt is provided the child of n with edge is updated, else n itself.
 * @param seq The sequence of the node to add data to.
 * @param s_opt: If provided the child of n with edge is updated, else if nullopt then n itself is updated.
 */
void overlap_fill_batch_wise::add_data_to_tree(unique_ptr<apta>& aut, const vector<int>& seq, const int reverse_type, float confidence, apta_node* node, const int symbol){
  static inputdata& id = *(inputdata_locator::get());
  
  if(confidence < 0.9){ // TODO: does this make sense?
    return;
  }

  trace* new_trace = active_learning_namespace::vector_to_trace(seq, id, reverse_type);
  id.add_trace_to_apta(new_trace, aut.get(), false);

  // the target should exist because we just added that state
  paul_data* data = dynamic_cast<paul_data*>(node->guard(symbol)->target->get_data());
  data->set_confidence(confidence);
}


/**
 * @brief Adds a child node to node with the edge labeled with symbol. The new node is automatically
 * filled with information from the sul (to avoid duplicate queries).
 * 
 * @param node The node.
 * @param symbol The symbol.
 */
/* void overlap_fill_batch_wise::add_child_node(unique_ptr<apta>& aut, apta_node* node, const int symbol){
  auto access_trace = node->get_access_trace();
  active_learning_namespace::pref_suf_t seq;
  seq = access_trace->get_input_sequence(true, true);
  seq[seq.size() - 1] = symbol;
            
  add_data_to_tree(aut, seq, sul, node, make_optional<int>(symbol));
}
 */

/**
 * @brief Here we collect all the traces we want to ask the transformer.
 * Side effect: Unknown types of nodes in between still get filled, just like in base_class. 
 */
void overlap_fill_batch_wise::pre_compute(vector< vector<int> >& query_traces, vector< pair<apta_node*, int> >& query_node_symbol_pairs, unordered_set<apta_node*>& seen_nodes, unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth){
  const static int max_search_depth = AL_MAX_SEARCH_DEPTH;
  if(max_search_depth > 0 && (left->get_depth() > max_search_depth || right->get_depth() > max_search_depth)) // making sure we don't bust the transformer
    return;
  
  else if(MAX_DEPTH > 0 && depth == MAX_DEPTH)
    return;
  
  else if(seen_nodes.contains(left) || seen_nodes.size() == 400) // TODO: REALLY BAD SOLUTION!!!
    return;
    
  seen_nodes.insert(left);
  auto r_data = dynamic_cast<paul_data*>(right->get_data());
  auto l_data = dynamic_cast<paul_data*>(left->get_data());

  if(r_data->label_is_queried() && l_data->label_is_queried()){
    return;
  }
  
  // in these two following if-clauses the side effect happens (see description)
  if(!r_data->has_type()){
    complete_node(right, aut);
  }
  if(!l_data->has_type()){
    complete_node(left, aut);
  }

  // first do the right side
  for(auto it = right->guards_start(); it != right->guards_end(); ++it){
    if(it->second->target == nullptr) continue; // no child with that guard

    int symbol = it->first;
    apta_guard* right_guard = it->second;
    apta_guard* left_guard = left->guard(symbol, right_guard);
      
    if(left_guard == nullptr || left_guard->target == nullptr){
      auto access_trace = left->get_access_trace();
      active_learning_namespace::pref_suf_t seq = access_trace->get_input_sequence(true, true);
      seq[seq.size()-1] = symbol;

      query_traces.push_back(move(seq));
      query_node_symbol_pairs.emplace_back(left, symbol);

      if(query_traces.size() == STREAMING_BATCH_SIZE){
        const sul_response response = sul->do_query(query_traces, *(inputdata_locator::get()));
        const vector<int>& answers = response.GET_INT_VEC();
        const vector<double>& confidences = response.GET_DOUBLE_VEC();

        for(int i=0; i < query_traces.size(); ++i){
          add_data_to_tree(aut, query_traces[i], answers[i], confidences[i], query_node_symbol_pairs[i].first, query_node_symbol_pairs[i].second);
        }

        query_traces.clear();
        query_node_symbol_pairs.clear();
      }
    } 
    else {
      apta_node* left_target = left_guard->target;
      apta_node* right_target = right_guard->target;
      
      apta_node* left_child = left_target->find();
      apta_node* right_child = right_target->find();
      
      if(left_child != right_child){
        pre_compute(query_traces, query_node_symbol_pairs, seen_nodes, aut, left_child, right_child, depth+1);
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
      auto access_trace = right->get_access_trace();
      active_learning_namespace::pref_suf_t seq = access_trace->get_input_sequence(true, true);
      seq[seq.size()-1] = symbol;

      query_traces.push_back(move(seq));
      query_node_symbol_pairs.emplace_back(right, symbol);

      if(query_traces.size() == STREAMING_BATCH_SIZE){
        const sul_response response = sul->do_query(query_traces, *(inputdata_locator::get()));
        const vector<int>& answers = response.GET_INT_VEC();
        const vector<double>& confidences = response.GET_DOUBLE_VEC();

        for(int i=0; i < query_traces.size(); ++i){
          add_data_to_tree(aut, query_traces[i], answers[i], confidences[i], query_node_symbol_pairs[i].first, query_node_symbol_pairs[i].second);
        }

        query_traces.clear();
        query_node_symbol_pairs.clear();
      }
    } 
    else {
      apta_node* left_target = left_guard->target;
      apta_node* right_target = right_guard->target;
      
      apta_node* left_child = left_target->find();
      apta_node* right_child = right_target->find();
      
      if(left_child != right_child){
        pre_compute(query_traces, query_node_symbol_pairs, seen_nodes, aut, left_child, right_child, depth+1);
      } 
    }
  }
}

/**
 * @brief Entry point to the overloaded function. Created a set of seen states and starts the main pre_compute subroutine.
 * 
 * For a more detailed description see the overloaded function.
 */
void overlap_fill_batch_wise::pre_compute(unique_ptr<apta>& aut, apta_node* left, apta_node* right){
  unordered_set<apta_node*> seen_nodes;
  vector< vector<int> > query_traces;
  vector< pair<apta_node*, int> > query_node_symbol_pairs;

  pre_compute(query_traces, query_node_symbol_pairs, seen_nodes, aut, left, right, 0);

  if(query_traces.size() == 0)
    return;

  // doing the remaining queries
  const sul_response response = sul->do_query(query_traces, *(inputdata_locator::get()));
  const vector<int>& answers = response.GET_INT_VEC();
  const vector<double>& confidences = response.GET_DOUBLE_VEC();  
  
  for(int i=0; i < query_traces.size(); ++i){
    add_data_to_tree(aut, query_traces[i], answers[i], confidences[i], query_node_symbol_pairs[i].first, query_node_symbol_pairs[i].second);
  }
}
