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
#include <set>
#include <string>

using namespace std;

void input_file_sul::post(){
    // this one does not need preprocessing and post yet, since we test it on abbadingo
}

void input_file_sul::step(){
    // this one does not need step() yet, since we test it on abbadingo
}

bool input_file_sul::is_member(const vector<int>& query_trace) const {
    return all_traces.contains(query_trace);
}

const int input_file_sul::query_trace(const vector<int>& query_trace, inputdata& id) const {
  if(!this->is_member(query_trace)){

    static bool added_unknown_type = false;
    static string unk_t_string = "unk";
    if(!added_unknown_type){
      int counter = 0;
      const auto& r_types = id.get_r_types();
      while(r_types.contains(unk_t_string)){
        unk_t_string = unk_t_string + to_string(counter);
      }

      id.add_unknown_type(unk_t_string);
      added_unknown_type = true;
    }

    return id.get_reverse_type(unk_t_string);
  }
  return all_traces.at(query_trace);
}

void input_file_sul::pre(inputdata& id){
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