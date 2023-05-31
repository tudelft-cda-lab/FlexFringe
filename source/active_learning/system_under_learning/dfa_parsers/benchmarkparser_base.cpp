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
#include "inputdatalocator.h"
#include "common_functions.h"

#include <fstream> 
#include <utility>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <functional>
#include <queue>
#include <unordered_set>

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
 * We build the graph successively breadth-first from the root node.
 * 
 * @param initial_state Initial state, needed to identify the root node.
 * @param edges state1, list< <state2, label> >.
 * @param nodes state, shape.
 * @return unique_ptr<apta> The sut. 
 */
unique_ptr<apta> benchmarkparser_base::construct_apta(const string_view initial_state, 
                                const unordered_map<string, list< pair<string, string> > >& edges,
                                const unordered_map<string, string>& nodes) const {

  unordered_map< string_view, list<trace*> > node_to_trace_map; // <s2, l> l = list of all traces leading to s2 
  unordered_map< string_view, apta_node* > string_to_node_map;

  queue< string_view > nodes_to_visit;
  unordered_set<string_view> visited_nodes;
  nodes_to_visit.push(initial_state);

  unique_ptr<apta> sut = unique_ptr<apta>(new apta());
  inputdata* id = inputdata_locator::get();

  apta_node* current_node = mem_store::create_node(nullptr);
  sut->root = current_node;
  int node_number = 0;

  string_to_node_map[initial_state] = sut->root;
  node_to_trace_map[initial_state] = list<trace*>();

  int sequence_nr = 0;
  int depth = 0;
  while(!nodes_to_visit.empty()){
    string_view s1 = nodes_to_visit.front();
    nodes_to_visit.pop();

    if(visited_nodes.contains(s1)) continue;

    current_node = string_to_node_map.at(s1);
    current_node->depth = depth;
    current_node->red = true;

    for(auto& node_label_pair: edges.at(string(s1)) ){
      auto& s2 = node_label_pair.first;
      auto& label = node_label_pair.second;

      tail* new_tail = mem_store::create_tail(nullptr);
      const int symbol = id->symbol_from_string(label);
      new_tail->td->symbol = symbol;

      trace* new_trace;
      if(s1 == initial_state){
        new_trace = mem_store::create_trace(id);
        new_trace->length = 1;

        new_tail->td->index = 0;
        new_trace->head = new_tail;
        new_trace->end_tail = new_tail;
      }
      else{
        trace* old_trace = node_to_trace_map.at(s1).front(); // TODO @Sicco: could we safely pick the access trace of the parent node? 
        new_trace = mem_store::create_trace(id, old_trace);
        new_trace->length = old_trace->length + 1;
        new_tail->td->index = old_trace->end_tail->td->index + 1;

        tail* old_end_tail = new_trace->end_tail;

        old_end_tail->future_tail = new_tail;
        new_tail->past_tail = old_end_tail;
        new_trace->end_tail = new_tail;
      }

      string_view type_str = nodes.at(static_cast<string>(s2));
      const int type = id->type_from_string(static_cast<string>(type_str));
      new_trace->type = type;
      new_trace->finalize();

      new_trace->sequence = ++sequence_nr;
      new_tail->tr = new_trace;

      apta_node* next_node;
      if(!string_to_node_map.contains(s2)){
        next_node = mem_store::create_node(nullptr);
        string_to_node_map[s2] = next_node;
        next_node->source = current_node;

        node_to_trace_map[s2] = list<trace*>();
      }
      else{
        next_node = string_to_node_map[s2];
      }
      current_node->set_child(symbol, next_node);
      current_node->add_tail(new_tail);
      current_node->data->add_tail(new_tail);

      // add the final probs as well
      next_node->add_tail(new_tail->future());
      next_node->data->add_tail(new_tail->future());

      id->add_trace(new_trace);
      node_to_trace_map.at(s2).push_back(new_trace);
      nodes_to_visit.push(s2);
    }
    visited_nodes.insert(s1);
    depth++;
  }

  return std::move(sut);
}