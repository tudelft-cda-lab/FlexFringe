/**
 * @file benchmarkparser_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "benchmarkparser_base.h"
#include "parameters.h"

#include <fstream> 
#include <utility>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <reference_wrapper>

using namespace std;
using namespace graph_information;

const bool DEBUG = false;

/**
 * @brief Reads a line of the input .dot file. 
 * 
 * IMPORTANT: Assumptions we make in this function:
 * 1. The last line of the .dot file that contains graph information
 * will only hold the closing braces }
 * 2. States will always be of shape (in order) state_name [shape="shape_type" label="label_name"]
 * 3. Transitions will always be of shape s1 -> s2[label="label_name"]. Note that
 * there is no whitespace in between s2 and the data.
 * 4. Initial transitions will always be prepended by __. That is __start0 [label="" shape="none"]
 * for the data and __start0 -> s1 for the transitions.
 * 5. The first line will just hold the opening of the graph information, 
 * but no relevant information about the graph itself.
 * 
 * @param input_stream 
 * @return unique_ptr<graph_base> 
 */
unique_ptr<graph_base> benchmarkparser_base::readline(ifstream& input_stream) const {
  string line, cell;
  std::getline(input_stream, line);

  if(line.empty()) return unique_ptr<graph_base>(nullptr);

  
  if(line.rfind('{') != std::string::npos)[[unlikely]]{
    return unique_ptr<graph_base>(new header_line());
  }  
  else if(line.rfind('}') != std::string::npos)[[unlikely]]{
    // test for correct formatting in block
    stringstream final_line(line);
    list<string> final_line_split;
    while (std::getline(final_line, cell, ' ')) final_line_split.push_back(std::move(cell));
    assert( ("Last line must be empy but the } character.", final_line_split.size() == 1));

    return unique_ptr<graph_base>(nullptr); // end of file
  }
  
  stringstream linesplit(line);
  vector<string> cells;
  while (std::getline(linesplit, cell, ' ')) cells.push_back(std::move(cell));

  unique_ptr<graph_base> res;
  if(cells.at(0).at(0) == '_' && cells.at(0).at(1) == '_'){
    // either an initial transition, e.g. " __start0 -> s1 " or a label of it, e.g. " __start0 [label="" shape="none"] "
    if(cells.at(1).compare("->") == 0){
      res = unique_ptr<initial_transition>(new initial_transition());
      dynamic_cast<initial_transition*>(res.get())->state = std::move(cells.at(2));
      dynamic_cast<initial_transition*>(res.get())->start_id = std::move(cells.at(0));
    }
    else{
      res = unique_ptr<initial_transition_information>(new initial_transition_information());
      dynamic_cast<initial_transition_information*>(res.get())->start_id = std::move(cells.at(0));

      string& symbol_ref = cells.at(1);
      const auto begin_idx_s = symbol_ref.find_first_of('\"') + 1;
      const auto end_idx_s = symbol_ref.find_last_of('\"');
      if(end_idx_s != begin_idx_s + 1){ // make sure that label is not empty
        const string symbol = symbol_ref.substr(begin_idx_s, end_idx_s - begin_idx_s - 1);
        dynamic_cast<initial_transition_information*>(res.get())->symbol = std::move(symbol);
      }

      // TODO: we'll possibly have to update this one in further cases down the road
      string& data_ref = cells.at(2);
      const auto begin_idx_d = data_ref.find_first_of('\"') + 1;
      const auto end_idx_d = data_ref.find_last_of('\"');
      if(end_idx_d != begin_idx_d + 1){
        const string data = data_ref.substr(begin_idx_d, end_idx_d - begin_idx_d);
        dynamic_cast<initial_transition_information*>(res.get())->data = std::move(data);
      }
    }
  }
  else if(cells.at(1).compare("->") == 0){
    // this is a transition
    res = unique_ptr<transition_element>(new transition_element());
    dynamic_cast<transition_element*>(res.get())->s1 = std::move(cells.at(0));
    
    string& s2_ref = cells.at(2);
    const auto pos_1 = s2_ref.find_first_of('[');
    const string s2 = s2_ref.substr(0, pos_1);
    dynamic_cast<transition_element*>(res.get())->s2 = std::move(s2);

    // note: no empty labels expected here
    const string label_str = "label=\""; 
    const auto label_pos = s2_ref.find(label_str, pos_1 + 1) + label_str.size();
    const auto label_end_pos = s2_ref.find_first_of('\"', label_pos);
    const string label = s2_ref.substr(label_pos, label_end_pos - label_pos);
    dynamic_cast<transition_element*>(res.get())->symbol = std::move(label);

    // TODO: we do not use data at this stage, yet
  }
  else if(cells.size()==3){
    res = unique_ptr<graph_node>(new graph_node());
    dynamic_cast<graph_node*>(res.get())->id = cells.at(0);

    // note: shape is expected to be never empty 
    const string shape_str = "shape=\"";
    string& data_ref = cells.at(1);
    const auto shape_pos = data_ref.find(shape_str, 1) + shape_str.size(); // 1 because starts with '['
    const auto shape_end_pos = data_ref.find_first_of('\"', shape_pos);
    const string shape = data_ref.substr(shape_pos, shape_end_pos - shape_pos);
    dynamic_cast<graph_node*>(res.get())->shape = std::move(shape);

    // TODO: we don't use the label of the node at this stage
  }
  else{
    throw logic_error("Wrong input file? Up until now only DFAs supported.");
  }

  return res;
}

/**
 * @brief Does what you think it does.
 * 
 * Works in two waves: First wave, we construct the nodes and the access traces and add them to the inputdata. 
 * In the second iteration, we add the data to the nodes.
 * 
 * @param initial_state Initial state, needed to identify the root node.
 * @param edges se
 * @param nodes se
 * @return unique_ptr<apta> The sut. 
 */
unique_ptr<apta> benchmarkparser_base::construct_apta(const string_view initial_state, 
                                const unordered_map<string, list< pair<string, string> > >& edges,
                                const unordered_map<string, string>& nodes) const {

  unique_ptr<apta> sut = unique_ptr<apta>(new apta());

  apta_node* current_node = mem_store::create_node(nullptr);
  sut->root = current_node;

  stack< reference_wrapper<string> > current_layer;
  current_layer.push(initial_state);

  stack< reference_wrapper<string> > next_layer;
  unordered_map< reference_wrapper<string>, apta_node*, active_learning_namespace::ref_wrapper_comparator<string> > id_to_node_map;
  id_to_node_map[initial_state] = current_node;
  unordered_set< reference_wrapper<string>, active_learning_namespace::ref_wrapper_comparator<string> > completed_states; // to deal with loops in automaton

  int depth = 0;
  int node_number = 0;

  while(true){
    while(!current_layer.empty()){
      string& s1 = current_layer.top();
      
      if(completed_states.count(s1) > 0) continue;

      current_node = id_to_node_map.at(s1);
      current_node->depth = depth;
      current_node->number = ++node_number;

      trace* current_access_trace = current_node->access_trace;

      for(auto& [s2, label]: edges.at(s1)){ // walking through the list of edges
        apta_node* next_node;
        if(id_to_node_map.count(s2) == 0){
          next_node = mem_store::create_node(nullptr);
          id_to_node_map[s2] = next_node;
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

  return sut;

}