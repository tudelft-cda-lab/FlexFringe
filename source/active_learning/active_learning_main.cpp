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

#include "lstar.h"
#include "lsharp.h"

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

const bool INPUT_FILE_SUL = true; // TODO: have better switch for that later

inputdata active_learning_namespace::get_inputdata(){

  bool read_csv = false;
  //ifstream input_stream = get_input_stream();
  
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

/*   for(const auto it: id){
    auto& current_trace = *it;
    const auto current_sequence = current_trace.get_input_sequence();
    all_traces.insert(current_sequence);
  } */
  input_stream.close();

  return id;
}

void active_learning_namespace::run_active_learning(){

  if(ACTIVE_LEARNING_ALGORITHM == "l_star"){
    inputdata id = get_inputdata();
    
    auto l_star = lstar_algorithm();
    l_star.run_l_star(id);
  }
  else if(ACTIVE_LEARNING_ALGORITHM == "l_sharp"){
    STORE_ACCESS_STRINGS = true;
    inputdata id = get_inputdata();

    auto l_sharp = lsharp_algorithm();
    l_sharp.run_l_sharp(id);
  }
  else{
    throw logic_error("Fatal error: Unknown active_learning_algorithm flag used: " + ACTIVE_LEARNING_ALGORITHM);
  }
}