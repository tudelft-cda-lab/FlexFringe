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

/**
 * @brief Adds the type to the node according to the SUL
 */
void ii_base::complete_node(apta_node* node, unique_ptr<apta>& aut){
  static inputdata& id = *(inputdata_locator::get());

  trace* access_trace = node->get_access_trace();
  active_learning_namespace::pref_suf_t seq;
  seq = access_trace->get_input_sequence(true, false);

  vector< vector<int> > query;
  query.push_back(move(seq));

  const sul_response response = sul->do_query(query, *(inputdata_locator::get()));
  const vector<int>& answers = response.GET_INT_VEC();
  const vector<double>& confidences = response.GET_DOUBLE_VEC();
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