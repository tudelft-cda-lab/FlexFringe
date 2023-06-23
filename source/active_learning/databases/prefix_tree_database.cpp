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

using namespace std;
using namespace active_learning_namespace;

virtual void prefix_tree_database::initialize(){
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
      id.read(&input_parser);
  } else {
      auto input_parser = abbadingoparser(input_stream);
      id.read(&input_parser);
  }

  the_tree = unique_ptr<apta>(new apta());
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
bool is_member(const std::list<int>& query_trace) const {
  trace* tr = active_learning_namespace::vector_to_trace(query_trace, inputdata_locator::get());
  return active_learning_namespace::aut_accepts_trace(trace* tr, apta* aut);
}