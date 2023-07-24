/**
 * @file nn_softmax_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * https://serizba.github.io/cppflow/quickstart.html#load-a-model
 * @version 0.1
 * @date 2023-07-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "nn_softmax_sul.h"

#include <cppflow/cppflow.h>

using namespace std;

bool nn_softmax_sul::is_member(const std::list<int>& query_trace) const {
  return true;
}

const int nn_softmax_sul::query_trace(const std::list<int>& query_trace, inputdata& id) const {
  
  auto output = model(input);

}

void nn_softmax_sul::pre(inputdata& id){
  model = cppflow::model("coolpredictor");
}
