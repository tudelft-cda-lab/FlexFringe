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
#include "mem_store.h"
#include "source/input/inputdatalocator.h"
#include "common_functions.h"

#include <string>
#include <sstream>
#include <stdexcept>
#include <list>
#include <utility>
#include <cassert>
#include <unordered_map>
#include <stack>
#include <functional>
#include <iostream>

using namespace std;
using namespace graph_information;

/**
 * @brief Read the input and construct a ready apta from it. 
 * 
 * TODO: We can possibly split the reading and the construction into two parts, making them more 
 * reusable and debugable.
 * 
 * @param input_stream 
 * @return unique_ptr<apta> 
 */
unique_ptr<apta> benchmark_dfaparser::read_input(ifstream& input_stream) const {
  string initial_state; // node_id
  unordered_map<string, list< pair<string, string> > > edges; // s1, list< <s2, label> >
  unordered_map<string, string> nodes; // s, shape

  unique_ptr<graph_base> line_info = readline(input_stream);
  while(line_info){
    if(dynamic_cast<transition_element*>(line_info.get()) != nullptr){
      //inputdata_locator::get()->type_from_string(i);
      //transition_element* li = dynamic_cast<transition_element*>(line_info.get());
      auto li_ptr = dynamic_cast<transition_element*>(line_info.get());
      if(!edges.contains(li_ptr->s1)){
        edges[li_ptr->s1] = list< pair<string, string> >();
      }
      edges[li_ptr->s1].push_back( make_pair( std::move(li_ptr->s2), std::move(li_ptr->symbol)) );
    }
    else if(dynamic_cast<graph_node*>(line_info.get()) != nullptr){
      //graph_node* gn = dynamic_cast<graph_node*>(line_info.get());
      auto li_ptr = dynamic_cast<graph_node*>(line_info.get());
      nodes[li_ptr->id] = std::move(li_ptr->shape);
    }
    else if(dynamic_cast<initial_transition*>(line_info.get()) != nullptr)[[unlikely]]{
      assert(("Only one initial state expected. Wrong input file?", initial_state.size() == 0));
      initial_state = std::move( dynamic_cast<initial_transition*>(line_info.get())->state );
    }
    else if(dynamic_cast<initial_transition_information*>(line_info.get()) != nullptr)[[unlikely]]{
      // Not needed for DFAs
      //inputdata_locator::get()->symbol_from_string(line_info->symbol);
      //initial_transition_information[line_info->start_id] = make_pair(line_info->symbol, line_info->data);
    }
    else if(dynamic_cast<header_line*>(line_info.get()) != nullptr)[[unlikely]]{
      // do nothing, but don't continue
    }
    else{
      throw logic_error("Unexpected object returned. Wrong input file?");
    }
    line_info = readline(input_stream);
  }

  auto sut = construct_apta(initial_state, edges, nodes);

  return std::move(sut);
}