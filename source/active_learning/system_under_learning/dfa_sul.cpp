/**
 * @file dfa_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "dfa_sul.h"
#include "parameters.h"

#include <fstream>
#include <stdexcept>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief This function does two tests: First it checks if there's an APTA file, and if yes reads it.
 * If this check fails it checks if the input file is json formatted, and if yes reads that one. Otherwise
 * throw an exception.
 * 
 * The reason for the design is that we can have multiple modes, either only active learning (then we 
 * use the INPUT file), or active and passive learning combined (then we use the APTA file).
 * 
 * @param id The input data (not used here).
 */
void dfa_sul::pre(inputdata& id) {
  ifstream input_apta_stream;

  if(!APTA_FILE.empty()){
    input_apta_stream = ifstream(APTA_FILE);
    cout << "Reading apta file (SUT) - " << APTA_FILE << endl;
  }
  else if (INPUT_FILE.compare(INPUT_FILE.length() - 5, INPUT_FILE.length(), ".json") == 0){
    input_apta_stream = ifstream(INPUT_FILE);
    cout << "Reading input file (SUT) - " << INPUT_FILE << endl;
  }
  else {
    throw logic_error("Require a json formatted apta file as an SUT");
  }

  sul->read_json(input_apta_stream);
}


bool dfa_sul::is_member(const std::vector<int>& query_trace) const {
  
}


const int dfa_sul::query_trace(const std::vector<int>& query_trace, inputdata& id) const {

}