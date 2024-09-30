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

void ii_base::complete_node(apta_node* node, std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher){
  static inputdata& id = *(inputdata_locator::get());

  trace* access_trace = node->get_access_trace();
  active_learning_namespace::pref_suf_t seq;
  seq = access_trace->get_input_sequence(true, false);

  vector< vector<int> > query;
  query.push_back(move(seq));

  vector< pair<string, float> > res = teacher->ask_type_confidence_batch(query, id);
  if(res.size() > 1)
    cerr << "Something weird happened in complete_node method of overlap_fill_batch_wise-class" << endl;
  
  string& type = res[0].first;
  float confidence = res[0].second;
  
  if(confidence < 0.9){
    return;
  }

  id.add_type(type);
  int reverse_type = id.get_reverse_type(type);

  trace* new_trace = active_learning_namespace::vector_to_trace(seq, id, reverse_type);
  id.add_trace_to_apta(new_trace, aut.get(), false);

  paul_data* data;
  data = dynamic_cast<paul_data*>(node->get_data());
  data->set_confidence(confidence);
}