/**
 * @file benchmark_dfaparser.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "benchmark_dfaparser.h"

#include <string>
#include <sstream>

using namespace std;

virtual std::optional<line_info> readline(ifstream& input_stream) const {
  string line, cell;
  std::getline(input_stream, line);
  if(line.empty()) return nullopt;

  line_info res;

  [[unlikely]]
  if(line.rfind("{") != line.begin()){
    res.is_header = true;
    return res;
  }

  stringstream linesplit(line);
  while (std::getline(ls2, cell, ' ')) {
    //row.push_back(cell);
    // make checks here
  }
}

unique_ptr<apta> benchmark_dfaparser::read_input(ifstream& input_stream) const {

  // TODO: don't forget a try-catch when reading a line, making sure that the format is correct and printing msg if not

  unique_ptr<apta> sut = unique_ptr<apta>(new apta());

  auto line = readline;
  while(line){
    // TODO: read line, construct automaton
  }


    
    vector<string> row;
    while (std::getline(ls2, cell, ',')) {
        row.push_back(cell);
    }
    std::getline(ls2, cell);
    row.push_back(cell);

    string id = "";
    for (auto i : id_cols) {
        if (!id.empty()) id.append("__");
        id.append(row[i]);
    }

    string type = "";
    for (auto i : type_cols) {
        if (!type.empty()) type.append("__");
        type.append(row[i]);
    }
    if(type.empty()) type = "0";

  return std::move(sut);
}