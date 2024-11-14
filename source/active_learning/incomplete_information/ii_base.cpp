/**
 * @file ii_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-08-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "ii_base.h"
#include "paul_heuristic.h"

#include "mem_store.h"
#include "inputdata.h"
#include "inputdatalocator.h"

#include <vector>
#include <utility>
#include <iostream>

#include "common_functions.h"

using namespace std;

void ii_base::complete_node(apta_node* node, std::unique_ptr<apta>& aut, std::unique_ptr<oracle_base>& oracle){
  static inputdata& id = *(inputdata_locator::get());

  trace* access_trace = node->get_access_trace();
  active_learning_namespace::pref_suf_t seq;
  seq = access_trace->get_input_sequence(true, false);

  vector< vector<int> > query;
  query.push_back(move(seq));

  const sul_response response = oracle->ask_sul(queries, *(inputdata_locator::get()));
  const vector<int>& answers = response.GET_INT_VEC();
  const vector<float>& confidences = response.GET_FLOAT_VEC();
  if(answers.size() > 1)
    cerr << "Something weird happened in complete_node method of overlap_fill_batch_wise-class" << endl;
  
  int reverse_type = answers[0];
  float confidence = confidences[0];

  trace* new_trace = active_learning_namespace::vector_to_trace(seq, id, reverse_type);
  id.add_trace_to_apta(new_trace, aut.get(), false);

  // TODO: make this one more generic. Too much on PAUL data at the moment
  paul_data* data;
  data = dynamic_cast<paul_data*>(node->get_data());
  data->set_confidence(confidence);
}

bool ii_base::check_consistency(std::unique_ptr<apta>& aut, std::unique_ptr<oracle_base>& oracle, apta_node* left, apta_node* right){
  static bool init = false;
  if(!init){
    init = true;
    std::cerr << "WARNING: This method does not support check_consistency() yet. Either change input parameters or implement this method." << std::endl;
  }

  return true;
}