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

void input_file_sul::parse_input(ifstream& input_stream, inputdata& id){
  bool read_csv = false;
  if(INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".csv") == 0){
      read_csv = true;
  }

  inputdata_locator::provide(&id);
  
  if(read_csv) {
      auto input_parser = csv_parser(input_stream, csv::CSVFormat().trim({' '}));
      id.read(&input_parser);
  } else {
      auto input_parser = abbadingoparser(input_stream);
      id.read(&input_parser);
  }

  for(const auto it: id){
    auto current_trace = **it;
    const auto current_sequence = current_trace->get_input_sequence();
    all_traces.insert(current_sequence);
  }
}

input_file_sul::input_file_sul() : parses_input_file(true){
}