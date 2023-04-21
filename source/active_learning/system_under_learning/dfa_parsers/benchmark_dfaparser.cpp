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

virtual unique_ptr<graph_base> readline(ifstream& input_stream) const {
  string line, cell;
  std::getline(input_stream, line);
  if(line.empty()) return nullopt;

  [[unlikely]]
  if(line.rfind('{') != line.begin()){
    return unique_ptr<graph_base>(new header_line());
  }  

  linetype lt = linetype::uninitialized;
  if(line.at(0) == '_' && line.at(1) == '_'){
    lt = linetype::initial_state;
  }
  //else if(line.at(0) == 's' && isdigit(linetype.at(1))){
  //  // TODO: not true yet
  //  lt = linetype::state;
  //}
  else if(line.rfind('}') != line.begin()){

    // test for correct formatting in block
    stringstream final_line(line);
    list<string> final_line_split;
    while (std::getline(final_line, cell, ' ')) final_line_split.push_back(std::move(cell));
    assert( (void) "Last line must be empy but the } char.", final_line_split.size() == 1);

    return unique_ptr<graph_base>(nullptr); // end of file
  }
  
  
  stringstream linesplit(line);
  list<string> cells;
  while (std::getline(linesplit, cell, ' ')) cells.push_back(std::move(cell));

  unique_ptr<graph_base> res;
  if(list.at(0).at(0) == '_' && list.at(0).at(1) == '_'){
    // either a initial transition, e.g. " __start0 -> s1 " or a label of it, e.g. " __start0 [label="" shape="none"] "
    if(list.at(1) == "->"){
      res = unique_ptr<graph_base>(new initial_transition());
      res->state = list.at(0);
    }
    else{
      // this is label
    }
  }
  return res;
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