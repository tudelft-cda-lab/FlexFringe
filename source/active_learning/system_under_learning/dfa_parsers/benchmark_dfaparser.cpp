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
  //map<string, pair<string, string> > initial_transition_information; // transition_id, <label, data> 
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
      inputdata_locator::get()->symbol_from_string(line_info->symbol);
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

  map<string, int> seen_nodes;
  map<int, apta_node*> int_to_node_map;

  seen_nodes[initial_state] = seen_nodes.size();
  apta_node* root = mem_store::create_node(nullptr);
  sut->get_root() = root;

  // TODO: we can do a stack for the nodes
  for (const auto& [id, shape] : nodes){
    const int node_number = seen_nodes.size();
    seen_nodes[id] = node_number;

    apta_node* new_node = mem_store::create_node(nullptr);
    new_node->set_number(node_number);
    int_to_node_map[node_number] = new_node;
  }

  // TODO: implement graph building efficiently
  for(const auto& [s1, s2_label_pair_list] : edges){
    const auto& s2 = s2_label_pair.first;
    const auto& label = s2_label_pair.second;


  }

  return std::move(sut);
}