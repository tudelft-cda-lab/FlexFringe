/**
 * @file common_functions.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "common_functions.h"

#include "abbadingoparser.h"
#include "csvparser.h"

#include <iostream>
#include <stdexcept>

using namespace std;
using namespace active_learning_namespace;

/* database create_db(){
  ifstream input_stream(INPUT_FILE);  
  cout << "Input file: " << INPUT_FILE << endl;
    
  if(!input_stream) {
      LOG_S(ERROR) << "Input file not found, aborting";
      std::cerr << "Input file not found, aborting" << std::endl;
      exit(-1);
  } else {
      std::cout << "Using input file: " << INPUT_FILE << std::endl;
  }

  bool read_csv = false;
  if(INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".csv") == 0){
      read_csv = true;
  }

  if(read_csv) {
      auto input_parser = csv_parser(input_stream, csv::CSVFormat().trim({' '}));
      throw logic_error("csv database not implemented yet.");
  } else {
      auto input_parser = abbadingoparser(input_stream);
      return abbadingo_database(input_parser);
  }
} */