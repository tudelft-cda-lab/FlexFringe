/**
 * @file input_file_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "parameters.h"
#include "input_file_sul.h"
#include "abbadingoparser.h"
#include "inputdatalocator.h"
#include "csvparser.h"

#include <iostream>

using namespace std;

void input_file_sul::preprocessing(){
    // this one does not need preprocessing and postprocessing yet, since we test it on abbadingo
}

void input_file_sul::postprocessing(){
    // this one does not need preprocessing and postprocessing yet, since we test it on abbadingo
}

void input_file_sul::step(){
    // this one does not need step() yet, since we test it on abbadingo
}

bool input_file_sul::is_member(const vector<int>& query_trace) const {
    return all_traces.contains(query_trace);
}

const int input_file_sul::query_trace(const vector<int>& query_trace) const {
  if(!this->is_member(query_trace)) return -1;
  return all_traces.at(query_trace);
}

void input_file_sul::parse_input(inputdata& id){
  for(const auto it: id){
    trace& current_trace = *it;
    const auto current_sequence = current_trace.get_input_sequence();
    if(all_traces.contains(current_sequence)) continue;

    const int type = current_trace.get_type();
    all_traces[current_sequence] = type;
  }
}

input_file_sul::input_file_sul() : sul_base(true){
}