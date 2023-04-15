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

#include "input_file_sul.h"
#include "input_file_oracle.h"
#include "dfa_sul.h"
#include "active_sul_oracle.h"

#include "parameters.h"
#include "inputdata.h"
#include "inputdatalocator.h"
#include "abbadingoparser.h"
#include "csvparser.h"
#include "main_helpers.h"

#include <stdexcept>
#include <fstream>
#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace std;
using namespace active_learning_namespace;

inputdata active_learning_main_func::get_inputdata() const {
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


/**
 * @brief Selects the SUL to be used.
 * 
 * @return shared_ptr<sul_base> The sul.
 */
shared_ptr<sul_base> active_learning_main_func::select_sul_class(const bool ACTIVE_SUL) const {
  if(ACTIVE_SUL){
    return shared_ptr<sul_base>(new dfa_sul());
  }
  return shared_ptr<sul_base>(new input_file_sul());
}

/**
 * @brief Selects the teacher to be used. In case alternative teachers want to be written.
 * 
 * @return unique_ptr<base_teacher> The teacher.
 */
unique_ptr<base_teacher> active_learning_main_func::select_teacher_class(shared_ptr<sul_base>& sul, const bool ACTIVE_SUL) const {
  return unique_ptr<base_teacher>( new base_teacher(sul) ); 
}

/**
 * @brief Selects the oracle to be used. In case alternative oracles want to be written.
 * 
 * @return unique_ptr<eq_oracle_base> The oracle.
 */
unique_ptr<eq_oracle_base> active_learning_main_func::select_oracle_class(shared_ptr<sul_base>& sul, const bool ACTIVE_SUL) const {
  if(ACTIVE_SUL){
    return unique_ptr<eq_oracle_base>( new active_sul_oracle(sul) );
  }
  return unique_ptr<eq_oracle_base>( new input_file_oracle(sul) );
}

/**
 * @brief Selects the parameters the algorithm runs with and runs the algorithm.
 * 
 */
void active_learning_main_func::run_active_learning(){
  assertm(ENSEMBLE_RUNS > 0, "nruns parameter must be larger than 0 for active learning.");

  const bool ACTIVE_SUL = (INPUT_FILE.compare(INPUT_FILE.length() - 5, INPUT_FILE.length(), ".json") == 0) ||
                    (INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".dot") == 0) ||
                    !APTA_FILE.empty();

  auto sul = select_sul_class(ACTIVE_SUL);
  auto teacher = select_teacher_class(sul, ACTIVE_SUL);
  auto oracle = select_oracle_class(sul, ACTIVE_SUL);

  unique_ptr<algorithm_base> algorithm;
  if(ACTIVE_LEARNING_ALGORITHM == "l_star"){
    algorithm = unique_ptr<algorithm_base>(new lstar_algorithm(sul, teacher, oracle));
  }
  else if(ACTIVE_LEARNING_ALGORITHM == "l_sharp"){
    STORE_ACCESS_STRINGS = true;
    algorithm = unique_ptr<algorithm_base>(new lsharp_algorithm(sul, teacher, oracle));
  }
  else{
    throw logic_error("Fatal error: Unknown active_learning_algorithm flag used: " + ACTIVE_LEARNING_ALGORITHM);
  }

  if(ACTIVE_SUL){
    // we do not want to run the input file
    inputdata id;
    inputdata_locator::provide(&id);
    
    sul->pre(id);
    algorithm->run(id);
  }
  else{
    // we only want to read the inputdata when we learn passively or from sequences
    inputdata id = get_inputdata();
    
    sul->pre(id);
    algorithm->run(id);
  }
}