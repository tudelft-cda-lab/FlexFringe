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

#include "parameters.h"
#include "inputdata.h"
#include "inputdatalocator.h"
#include "abbadingoparser.h"
#include "csvparser.h"
#include "main_helpers.h"

#include <stdexcept>
#include <fstream>

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
shared_ptr<sul_base> active_learning_main_func::select_sul_class() const {
  return shared_ptr<input_file_sul>(new input_file_sul());
}

/**
 * @brief Selects the teacher to be used. In case alternative teachers want to be written.
 * 
 * @return unique_ptr<base_teacher> The teacher.
 */
unique_ptr<base_teacher> active_learning_main_func::select_teacher_class(shared_ptr<sul_base>& sul) const {
  return unique_ptr<base_teacher>( new base_teacher(sul.get()) ); 
}

/**
 * @brief Selects the oracle to be used. In case alternative oracles want to be written.
 * 
 * @return unique_ptr<eq_oracle_base> The oracle.
 */
unique_ptr<eq_oracle_base> active_learning_main_func::select_oracle_class(shared_ptr<sul_base>& sul) const {
  return unique_ptr<eq_oracle_base>( new input_file_oracle(sul) );
}

/**
 * @brief Selects the parameters the algorithm runs with and runs the algorithm.
 * 
 */
void active_learning_main_func::run_active_learning(){

  auto sul = select_sul_class();
  auto teacher = select_teacher_class(sul);
  auto oracle = select_oracle_class(sul);

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
  if(false){
    // TODO: make a check for the type of SUL you got
    // we do not want to run the input file
    algorithm->run(inputdata());
  }
  else{
    // we only want to read the inputdata when we learn passively or from sequences
    algorithm->run(get_inputdata());
  }
}