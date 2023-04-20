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

using namespace std;

const bool DEBUG = false;

unique_ptr<apta> benchmarkparser_base::get_apta() const {
  ifstream input_apta_stream;
  assert((void) "Input must be .dot formatted.", 
    INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".dot") == 0 ||
    APTA_FILE.compare(INPUT_FILE.length() - 4, APTA_FILE.length(), ".dot") == 0);

  // read_json stores alphabet and types in inputdata
  if(!APTA_FILE.empty()){
    input_apta_stream = ifstream(APTA_FILE);
  }
  else{
    input_apta_stream = ifstream(INPUT_FILE);
  }

  // TODO: parse the input file now
  unique_ptr<apta> sut = read_input(input_apta_stream);

  if(DEBUG){
    ofstream output("test.dot");
    stringstream aut_stream;
    sut->print_dot(aut_stream);
    output << aut_stream.str();
    output.close();
  }

  return std::move(sut);
}