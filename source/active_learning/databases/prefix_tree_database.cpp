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

using namespace std;

virtual void prefix_tree_database::initialize(){
  // open stream, build the apta
  bool read_csv = false;
  if(APTA_FILE.compare(APTA_FILE.length() - 4, APTA_FILE.length(), ".csv") == 0){
      read_csv = true;
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
  input_stream.close();
  return id;
}