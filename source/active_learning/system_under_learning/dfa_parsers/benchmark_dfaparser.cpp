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

using namespace std;
using namespace graph_information;

/**
 * @brief Read the input, returns a ready apta. 
 * 
 * Node: This function is definitely not optimized on an algorithmic level. You can do that later, but it should not
 * matter that much, since we only call it once, and it should still be relatively cheap.
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

    [[unlikely]]
    if(dynamic_cast<initial_transition*>(line_info.get()) != nullptr){
      assert(("Only one initial state expected. Wrong input file?", initial_state.size() == 0));
      initial_state = line_info->state;
    }
    else if(dynamic_cast<initial_transition_information*>(line_info.get()) != nullptr){
      // Not needed for DFAs
      //inputdata_locator::get()->symbol_from_string(line_info->symbol);
      //initial_transition_information[line_info->start_id] = make_pair(line_info->symbol, line_info->data);
    }
    else if(dynamic_cast<transition_element*>(line_info.get()) != nullptr){
      //inputdata_locator::get()->type_from_string(i);
      //transition_element* li = dynamic_cast<transition_element*>(line_info.get());
      if(!edges.contains(line_info->s1)){
        edges[line_info->s1] = list< pair<string, string> >;
      }
      edges[line_info->s1] = push_back( make_pair(line_info->s2, line_info->symbol) );
    }
    else if(dynamic_cast<graph_node*>(line_info.get()) != nullptr){
      //graph_node* gn = dynamic_cast<graph_node*>(line_info.get());
      nodes[line_info->id] = line_info->shape;
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
  map< reference_wrapper<string>, apta_node* > id_to_node_map;
  id_to_node_map[initial_state] = current_node;
  set< reference_wrapper<string> > completed_states; // to deal with loops in automaton

  int depth = 0;
  int node_number = 0;
  while(true){
    while(!current_layer.empty()){
      const string& s1 = current_layer.top();
      if(completed_states.contains(s1)) continue;

      current_node = id_to_node_map.at(s1);
      current_node->depth = depth;
      current_node->number = node_number;

      for(const auto& [s2, label]: edges.at(s1)){ // walking through the list of edges
        apta_node* next_node;
        if(!id_to_node_map.contains(s2)){
          next_node = mem_store::create_node(nullptr);
          id_to_node_map[s2] = next_node;
          next_node->number = ++node_number;
        }
        else{
          next_node = id_to_node_map[s2];
        }
        
        const int symbol = inputdata_locator::get()->symbol_from_string(label);
        current_node->set_child(symbol, next_node);

        // question: my approach works for trees. What about state machines?
        if(!completed_states.contains(s2)) next_layer.push(s2);
        // TODO: squeeze in the typing here as well in the form of traces
      }

      current_layer.pop();
      completed_states.insert(s1);
    }

    ++depth;
    if(next_layer.empty()) break;

    current_layer = std::move(next_layer);
    next_layer = stack< reference_wrapper<string> >;
  }

  return std::move(sut);
}