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
#include "ctype.h"

#include <string>
#include <sstream>
#include <stdexcept>
#include <list>
#include <utility>
#include <cassert>

using namespace std;

unique_ptr<apta> benchmark_dfaparser::read_input(ifstream& input_stream) const {

  unique_ptr<apta> sut = unique_ptr<apta>(new apta());
  unique_ptr<graph_information::graph_base>  line_info = readline(input_stream);
  while(line_info){
     line_info = readline(input_stream);
  }


    
/*     vector<string> row;
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
    if(type.empty()) type = "0"; */

  return std::move(sut);
}