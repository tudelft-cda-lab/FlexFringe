/**
 * @file factories.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "factories.h"

// DFAs
#include "lsharp.h"
#include "lstar.h"
#include "lstar_imat.h"

// weighted machines
#include "probabilistic_lsharp.h"
#include "weighted_lsharp.h"

// databases etc.
#include "paul.h"
//#include "ldot.h"

// the SULs
#include "database_sul.h"
#include "dfa_sul.h"
#include "input_file_sul.h"
//#include "sqldb_sul.h"

// the neural network SULs
#include "nn_binary_output_sul.h"
#include "nn_discrete_and_float_output_sul.h"
#include "nn_discrete_output_and_hidden_reps_sul.h"
#include "nn_discrete_output_sul.h"
#include "nn_float_output_sul.h"
#include "nn_float_vector_output_sul.h"

// the oracles
#include "input_file_oracle.h"
#include "discrete_output_sul_oracle.h"
//#include "sqldb_sul_random_oracle.h"
//#include "sqldb_sul_regex_oracle.h"
#include "paul_oracle.h"

// the ii_handlers
#include "distinguishing_sequence_fill.h"
#include "distinguishing_sequence_fill_fast.h"
#include "overlap_fill_batch_wise.h"
#include "overlap_fill.h"

// parsers and input-data representations
#include "abbadingoparser.h"
#include "csvparser.h"
#include "input/abbadingoreader.h"
#include "inputdata.h"
#include "inputdatalocator.h"
#include "common.h"
#include "parameters.h"

#include <initializer_list>

using namespace std;

/**
 * @brief Gets the SUl and returns it.
 */
shared_ptr<sul_base> sul_factory::select_sul(string_view sul_name){
  // non-neural network suls
  if(sul_name=="input_file_sul")
    return make_shared<input_file_sul>();
  else if(sul_name=="dfa_sul")
    return make_shared<dfa_sul>();
  else if(sul_name=="database_sul")
    return make_shared<database_sul>();
  else if(sul_name=="sqldb_sul"){
    throw invalid_argument("sqldb_sul currently commented out in sul_factory");//return make_shared<sqldb_sul>();
    // HIELKE: The code for these functions was in main.cpp, currently commented out
/*     bool LOADSQLDB = false;
    if (!LOADSQLDB) {
        // If reading, not loading, from db, do not drop on initialization.
        POSTGRESQL_DROPTBLS = false;
    }
    auto my_sqldb = make_unique<psql::db>(POSTGRESQL_TBLNAME, POSTGRESQL_CONNSTRING);
    if (LOADSQLDB) {
        ifstream input_stream = get_inputstream();

        cout << "Selected to use the SQL database. Creating new inputdata object and loading traces."
        abbadingo_inputdata id;
        inputdata_locator::provide(&id);
        my_sqldb->load_traces(id, input_stream);
        
        return my_sqldb;
    } */
  }

  // the neural network suls
  else if(sul_name=="nn_binary_output_sul")
    return make_shared<nn_binary_output_sul>();
  else if(sul_name=="nn_discrete_and_float_output_sul")
    return make_shared<nn_discrete_and_float_output_sul>();
  else if(sul_name=="nn_discrete_output_and_hidden_reps_sul")
    return make_shared<nn_discrete_output_and_hidden_reps_sul>();
  else if(sul_name=="nn_discrete_output_sul")
    return make_shared<nn_discrete_output_sul>();
  else if(sul_name=="nn_float_output_sul")
    return make_shared<nn_float_output_sul>();
  else if(sul_name=="nn_float_vector_output_sul")
    return make_shared<nn_float_vector_output_sul>();
  else
    throw invalid_argument("Input parameter specifying system under learning has been invalid. Please check your input.");
}

/**
 * @brief SUL is based only on the input parameters. Returns one or more SULs, based on how many are desired.
 */
vector< shared_ptr<sul_base> > sul_factory::create_suls(){
  if(AL_SYSTEM_UNDER_LEARNING.size() == 0)
    throw logic_error("System under learning must be specified. Aborting program.");

  vector< shared_ptr<sul_base> > res;
  res.push_back(select_sul(AL_SYSTEM_UNDER_LEARNING));
  
  if(AL_SYSTEM_UNDER_LEARNING_2.size() > 0 && AL_SYSTEM_UNDER_LEARNING != AL_SYSTEM_UNDER_LEARNING_2)
    res.push_back(select_sul(AL_SYSTEM_UNDER_LEARNING_2));
  else if (AL_SYSTEM_UNDER_LEARNING_2.size() > 0)
    res.push_back(res[0]); // we do a copy of the first shared pointer

  inputdata* id = inputdata_locator::get();
  if(id == nullptr)
    throw logic_error("Inputdata must exist the moment the SUL is created");
    
  for(auto& sul: res)
    sul->pre(*id);

  return res;
}

/**
 * @brief Creates the incomplete-information handler.
 */
shared_ptr<ii_base> ii_handler_factory::create_ii_handler(const shared_ptr<sul_base>& sul, string_view ii_name){
  if(ii_name.size() == 0)
    return shared_ptr<ii_base>(nullptr);
  else if(ii_name == "distinguishing_sequence_fill")
    return make_shared<distinguishing_sequence_fill>(sul);
  else if(ii_name == "distinguishing_sequence_fill_fast")
    return make_shared<distinguishing_sequence_fill_fast>(sul);
  else if(ii_name == "overlap_fill_batch_wise")
    return make_shared<overlap_fill_batch_wise>(sul);
  else if(ii_name == "overlap_fill")
    return make_shared<overlap_fill>(sul);
  else
    throw logic_error("Invalid ii_handler name");
}

/**
 * @brief Does what you think it does.
 */
unique_ptr<oracle_base> oracle_factory::create_oracle(const shared_ptr<sul_base>& sul, string_view oracle_name, const shared_ptr<ii_base>& ii_handler){
  if(AL_ORACLE == "discrete_output_sul_oracle")
      return make_unique<discrete_output_sul_oracle>(sul);
  
  if(AL_ORACLE == "input_file_oracle")
      return make_unique<input_file_oracle>(sul);
  else if(AL_ORACLE == "paul_oracle")
      return make_unique<paul_oracle>(sul, ii_handler);
  else if(AL_ORACLE == "sqldb_sul_random_oracle")
      throw std::runtime_error("Invalid oracle: Not included at the moment");
      //return make_unique<sqldb_sul_random_oracle>(sul);
  else if(AL_ORACLE == "sqldb_sul_regex_oracle")
      throw std::logic_error("Not implemented yet");
      //return make_unique<sqldb_sul_regex_oracle>(sul);
  else if(AL_ORACLE == "string_probability_oracle")
      return make_unique<string_probability_oracle>(sul);
  else
    throw std::invalid_argument("One of the oracle specifying input parameters was not recognized by the oracle factory.");
}


/**
 * @brief Does what you think it does.
 */
unique_ptr<algorithm_base> algorithm_factory::create_algorithm_obj(){
  vector< shared_ptr<sul_base> > sul_vec = sul_factory::create_suls(sul_factory::sul_key());
  assert(sul_vec.size() > 0 && sul_vec.size() <= 2);

  shared_ptr<ii_base> ii_handler = ii_handler_factory::create_ii_handler(sul_vec[0], AL_II_NAME, ii_handler_factory::ii_handler_key());
  if(ii_handler)
      cout << "ii_handler name has been provided. It is being equipped with the SUL named in the first sul name." << endl;
  
  unique_ptr<algorithm_base> res;
  if(sul_vec.size()==1){
    unique_ptr<oracle_base> oracle_1 = oracle_factory::create_oracle(sul_vec[0], AL_ORACLE, ii_handler, oracle_factory::oracle_key());
    res = create_algorithm_obj(move(oracle_1), ii_handler);
  }
  else if(sul_vec.size()==2){
    unique_ptr<oracle_base> oracle_1 = oracle_factory::create_oracle(sul_vec[0], AL_ORACLE, ii_handler, oracle_factory::oracle_key());
    unique_ptr<oracle_base> oracle_2 = oracle_factory::create_oracle(sul_vec[1], AL_ORACLE_2, ii_handler, oracle_factory::oracle_key());
    vector< unique_ptr<oracle_base> > oracles(2);
    oracles[0] = move(oracle_1);
    oracles[1] = move(oracle_2);
    res = create_algorithm_obj(move(oracles), ii_handler);
  }

  assert(res); // nullptr check
  return res;
}