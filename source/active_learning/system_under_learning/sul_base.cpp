/**
 * @file sul_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Implementation of SUL base class.
 * @version 0.1
 * @date 2023-02-19
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "sul_base.h"
#include "parameters.h"

#include <iostream>
#include <stdexcept>

using namespace std;

ifstream sul_base::get_input_stream() const {
  if(!parses_input_file){
    throw logic_error("Cannot get input stream of SUL-class not supporting it.");
  }

  ifstream input_stream(INPUT_FILE);  
  cout << "Input file: " << INPUT_FILE << endl;
    
  if(!input_stream) {
      cerr << "Input file not found, aborting" << endl;
      exit(-1);
  } else {
      cout << "Using input file: " << INPUT_FILE << endl;
  }
  return input_stream;
}