/**
 * @file prefix_tree_database.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-06-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "prefix_tree_database.h"
#include "common_functions.h"
#include "parameters.h"
#include "input/inputdatalocator.h"
#include "input/parsers/csvparser.h"
#include "source/input/parsers/abbadingoparser.h"
#include "common_functions.h"
#include "trace.h"
#include "inputdatalocator.h"

#include <stack>

using namespace std;
using namespace active_learning_namespace;

void prefix_tree_database::initialize(){
  bool read_csv = false;
  if(APTA_FILE.compare(APTA_FILE.length() - 4, APTA_FILE.length(), ".csv") == 0){
      read_csv = true;
  }

  if(read_csv && INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".csv") != 0){
    throw std::runtime_error("Error: Input and aptafile must be formatted the same to map to the same alphabet. Terminating.");
  }

  ifstream input_stream(APTA_FILE);  
  cout << "Apta-file: " << APTA_FILE << endl;
  if(!input_stream) {
      cerr << "Apta-file not found, aborting" << endl;
      exit(-1);
  } else {
      cout << "Using Apta-file: " << APTA_FILE << endl;
  }

  auto id = inputdata_locator::get();
  if(read_csv) {
      auto input_parser = csv_parser(input_stream, csv::CSVFormat().trim({' '}));
      id->read(&input_parser);
  } else {
      auto input_parser = abbadingoparser(input_stream);
      id->read(&input_parser);
  }

  the_tree = make_unique<apta>();
  id->add_traces_to_apta(the_tree.get());

  input_stream.close();
  id->clear_traces();
}

/**
 * @brief Checks if trace is in database. Type does not matter, as 
 * it is simply a "is in a set" query.
 * 
 * @param query_trace The trace we query.
 */
bool prefix_tree_database::is_member(const std::list<int>& query_trace) const {
  trace* tr = active_learning_namespace::vector_to_trace(query_trace, *(inputdata_locator::get()));
  return active_learning_namespace::aut_accepts_trace(tr, the_tree.get());
}

/**
 * @brief Takes in a node, and updates its statistics accordingly. 
 * Statistics are feature of the currently used merge-check.
 * 
 * @param n The node to be updated.
 */
void prefix_tree_database::update_state_with_statistics(apta_node* n){
  trace* access_trace = n->get_access_trace();
  //if(access_trace->end_tail->is_final()){
  //  access_trace->end_tail = access_trace->end_tail->past();
  //}

  apta_node* n_db = the_tree->sift(access_trace);

  list< pair<trace*, int> > trace_count_list = extract_tails_from_tree(n_db);
  for(auto& [tr, count]: trace_count_list){
    for(int i = 0; i < count; ++i){
      n->add_tail(tr->head);
      n->get_data()->add_tail(tr->head);
    }
  }
}

/**
 * @brief Performs a DFS search through the apta starting from node start and 
 * returns a list of the tails including the fitting statistics to update the 
 * nodes and tail_data of the nodes of the starting node.
 * 
 * Sidenote: I did implement this in an iterative manner, because the trees can
 * potentially become very large.
 * 
 * @param start The node to update. We start here.
 * @return std::list< std::unique_ptr<tail> > List of appropriately set tails.
 */
list< pair<trace*, int> > prefix_tree_database::extract_tails_from_tree(apta_node* start) {
  list< pair<trace*, int> > res; // we use traces for their copy constructor

  apta_node* current_node = start;
  trace* current_trace = nullptr;

  stack< pair<apta_node*, trace*> > nodes_to_traverse;
  unordered_set<int> visited_nodes;

  do {
    auto symbols = current_node->get_all_transition_symbols();

    if(symbols.empty()){
      // we hit a leaf node, pull backwards
      current_trace->finalize();
      res.push_back( make_pair(current_trace, current_node->get_size()) );
    } 
    if(visited_nodes.contains(current_node->get_number()) ){
      // a step backwards through the DFS
      int next_counts = 0;
      for(auto symbol: symbols){ 
        next_counts += current_node->get_child(symbol)->get_size();
      }

      current_trace->finalize();
      res.push_back( make_pair( current_trace, current_node->get_size() - next_counts));
    }
    else{
      for(auto symbol: symbols){ 
        trace* new_trace = mem_store::create_trace(inputdata_locator::get(), current_trace);
        if(current_trace == nullptr){
          [[unlikely]]
          tail* new_tail = mem_store::create_tail(nullptr);
          active_learning_namespace::update_tail(new_tail, symbol);

          new_trace->length = 1;
          new_tail->tr = new_trace;
          new_trace->head = new_tail;
          new_trace->end_tail = new_tail;
        }
        else{
          tail* old_tail = new_trace->end_tail;
          tail* new_tail = mem_store::create_tail(nullptr);
          active_learning_namespace::update_tail(new_tail, symbol);
          old_tail->future_tail = new_tail;
          new_tail->past_tail = old_tail; 

          new_trace->length = current_trace->get_length()+1;
          new_tail->tr = new_trace;
          new_trace->end_tail = new_tail;
        }
        nodes_to_traverse.push( make_pair(current_node->get_child(symbol), new_trace) );
      }
    }

    current_node = nodes_to_traverse.top().first;
    current_trace = nodes_to_traverse.top().second;
    visited_nodes.insert(current_node->get_number());

    nodes_to_traverse.pop();
  } while(!nodes_to_traverse.empty());
  
  return res;
}