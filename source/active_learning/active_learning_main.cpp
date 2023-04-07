/**
 * @file active_learning_main.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The main subroutine that is starting the active learning.
 * @version 0.1
 * @date 2023-03-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "active_learning_main.h"

#include "algorithm_base.h"
#include "lstar.h"
#include "lsharp.h"

#include "sul_base.h"
#include "input_file_sul.h"

#include "parameters.h"
#include "inputdata.h"
#include "inputdatalocator.h"
#include "abbadingoparser.h"
#include "csvparser.h"
#include "main_helpers.h"

#include <stdexcept>
#include <fstream>
#include <memory>

using namespace std;
using namespace active_learning_namespace;

inputdata active_learning_namespace::get_inputdata(){
  bool read_csv = false;
  if(INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".csv") == 0){
      read_csv = true;
  }

  ifstream input_stream(INPUT_FILE);  
  cout << "Input file: " << INPUT_FILE << endl;
  if(!input_stream) {
      cerr << "Input file not found, aborting" << endl;
      exit(-1);
  } else {
      cout << "Using input file: " << INPUT_FILE << endl;
  }

  inputdata id;
  inputdata_locator::provide(&id);
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

void active_learning_namespace::run_active_learning(){
/*   inputdata id; // = get_inputdata();
  cout << "original addres" << &id << endl;

  if(INPUT_FILE.compare(INPUT_FILE.length() - 5, INPUT_FILE.length(), ".json") != 0){
    cout << "Reading input data" << endl;
    id = get_inputdata();
    cout << "new address" << &id << endl;
  }
  else{
    cout << "Found neither abbadingo formatted file nor .csv formatted file. Treating provided input as SUT." << endl;
  } */

  unique_ptr<sul_base> sul = unique_ptr<input_file_sul>(new input_file_sul());

  unique_ptr<algorithm_base> algorithm;
  if(ACTIVE_LEARNING_ALGORITHM == "l_star"){
    algorithm = unique_ptr<algorithm_base>(new lstar_algorithm(sul));
  }
  else if(ACTIVE_LEARNING_ALGORITHM == "l_sharp"){
    STORE_ACCESS_STRINGS = true;
    algorithm = unique_ptr<algorithm_base>(new lsharp_algorithm(sul));
  }
  else{
    throw logic_error("Fatal error: Unknown active_learning_algorithm flag used: " + ACTIVE_LEARNING_ALGORITHM);
  }
  if(false){
    // TODO
  }
  else{
    algorithm->run(get_inputdata());
  }
}