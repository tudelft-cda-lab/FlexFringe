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
 * @brief Adds the type to the node according to the SUL and heuristic.
 * Only works on a few selected heuristics that support adding data.
 */
void ii_base::complete_node(apta_node* node, unique_ptr<apta>& aut){
  static inputdata& id = *(inputdata_locator::get());
  using ht = heuristic_type;
  
  static auto query_node = [this](apta_node* n, const unique_ptr<apta>& aut){
    trace* access_trace = n->get_access_trace();
    active_learning_namespace::pref_suf_t seq;
    seq = access_trace->get_input_sequence(true, false);

    vector< vector<int> > query;
    query.push_back(move(seq));

    return sul->do_query(query, *(inputdata_locator::get()));
  };

  static heuristic_type heuristic_name = ht::UNINITIALIZED;

  if(heuristic_name==ht::UNINITIALIZED) [[unlikely]] {

    auto determine_heuristic_type = [](apta_node* n){
      if(dynamic_cast<paul_data*>(n->get_data())){
        return ht::PAUL_H;
      }
      else{
        return ht::OTHER;
      }
    };

    heuristic_name = determine_heuristic_type(node);
  }
  else if(heuristic_name == ht::PAUL_H){
      auto resp = query_node(node, aut);

      const vector<int>& answers = resp.GET_INT_VEC();
      const vector<double>& confidences = resp.GET_DOUBLE_VEC();
      if(answers.size() > 1)
        cerr << "Something weird happened in complete_node method of overlap_fill_batch_wise-class" << endl;

      int reverse_type = answers[0];
      float confidence = confidences[0];

      paul_data* data = dynamic_cast<paul_data*>(node->get_data());
      data->set_confidence(confidence);
      data->add_inferred_type(reverse_type);
  }
  else{
      static bool has_notified = false;
      if(!has_notified){
        cout << "Chosen heuristic does not support adding inferred data to nodes. Omitting this step." << endl;
        has_notified = true;
      }
      return;
  }
}