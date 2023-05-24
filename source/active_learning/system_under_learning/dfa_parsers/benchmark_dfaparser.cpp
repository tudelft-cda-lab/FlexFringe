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
#include <map>
#include <set>
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
  map<string, list< pair<string, string> > > edges; // s1, list< <s2, label> >
  map<string, string> nodes; // s, shape

  unique_ptr<graph_base> line_info = readline(input_stream);
  while(line_info){
    if(dynamic_cast<transition_element*>(line_info.get()) != nullptr){
      //inputdata_locator::get()->type_from_string(i);
      //transition_element* li = dynamic_cast<transition_element*>(line_info.get());
      auto li_ptr = dynamic_cast<transition_element*>(line_info.get());
      if(!edges.contains(li_ptr->s1)){
        edges[li_ptr->s1] = list< pair<string, string> >();
      }
      edges[li_ptr->s1].push_back( make_pair(li_ptr->s2, li_ptr->symbol) );
    }
    else if(dynamic_cast<graph_node*>(line_info.get()) != nullptr){
      //graph_node* gn = dynamic_cast<graph_node*>(line_info.get());
      auto li_ptr = dynamic_cast<graph_node*>(line_info.get());
      nodes[li_ptr->id] = li_ptr->shape;
    }
    else if(dynamic_cast<initial_transition*>(line_info.get()) != nullptr)[[unlikely]]{
      assert(("Only one initial state expected. Wrong input file?", initial_state.size() == 0));
      initial_state = dynamic_cast<initial_transition*>(line_info.get())->state;
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

  unique_ptr<apta> sut = unique_ptr<apta>(new apta());

  apta_node* current_node = mem_store::create_node(nullptr);
  sut->root = current_node;

  stack< reference_wrapper<string> > current_layer;
  current_layer.push(initial_state);

  stack< reference_wrapper<string> > next_layer;
  map< reference_wrapper<string>, apta_node*, active_learning_namespace::ref_wrapper_comparator<string> > id_to_node_map;
  id_to_node_map[initial_state] = current_node;
  set< reference_wrapper<string>, active_learning_namespace::ref_wrapper_comparator<string> > completed_states; // to deal with loops in automaton

  int depth = 0;
  int node_number = 0;

  while(true){
    while(!current_layer.empty()){
      string& s1 = current_layer.top();
      
      if(completed_states.count(s1) > 0) continue;

      current_node = id_to_node_map.at(s1);
      current_node->depth = depth;
      current_node->number = node_number;

      trace* current_access_trace = current_node->access_trace;

      for(auto& [s2, label]: edges.at(s1)){ // walking through the list of edges
        apta_node* next_node;
        if(id_to_node_map.count(s2) == 0){
          next_node = mem_store::create_node(nullptr);
          id_to_node_map[s2] = next_node;
          next_node->number = ++node_number;
        }
        else{
          next_node = id_to_node_map[s2];
        }
        
        const int symbol = inputdata_locator::get()->symbol_from_string(label);
        current_node->set_child(symbol, next_node);

        trace* new_access_trace;
        tail* new_tail = mem_store::create_tail(nullptr);
        active_learning_namespace::update_tail(new_tail, symbol);
        
        if(depth > 0){
          // the root node has no valid access trace
          new_access_trace = mem_store::create_trace(inputdata_locator::get(), current_access_trace); 
          new_tail->td->index = current_access_trace->end_tail->td->index + 1;
        } 
        else [[unlikely]] {
          new_access_trace = mem_store::create_trace();
          new_tail->td->index = 1; // TODO: start from 0 or 1?
        } 

        new_access_trace->end_tail = new_tail;
        new_access_trace->finalize();
        next_node->access_trace = new_access_trace;

        // question: my approach works for trees. What about state machines?
        if(completed_states.count(s2) == 0) next_layer.push(s2);
      }

      current_layer.pop();
      completed_states.insert(s1);
    }

    ++depth;
    if(next_layer.empty()) break;

    current_layer = std::move(next_layer);
    next_layer = stack< reference_wrapper<string> >();
  }

  return std::move(sut);
}