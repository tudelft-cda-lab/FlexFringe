/**
 * @file input_file_oracle.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-03-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "input_file_oracle.h"
#include "predict.h"
#include "common_functions.h"

using namespace std;
using namespace active_learning_namespace;

bool input_file_oracle::apta_accepts_trace(state_merger* merger, const vector<int>& tr) const {
  const static double THRESH = 1; // TODO: we need to set this guy somehow

  trace* new_tr = vector_to_trace(tr);
  double score = predict_trace(merger, new_tr);
  mem_store::delete_trace(new_tr);

  return score < THRESH;
}

optional< vector<int> > input_file_oracle::equivalence_query(state_merger* merger) {
  for(const auto& tr: sul->get_all_traces()){
    if(!apta_accepts_trace(aut, tr)){
      return make_optional< vector<int> >(tr);
    }
  }
  return nullopt; // empty optional
}