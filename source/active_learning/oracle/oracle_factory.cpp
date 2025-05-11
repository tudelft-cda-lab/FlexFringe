/**
 * @file oracle_factory.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-05-10
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "oracle_factory.h"

#include "input_file_oracle.h"
#include "paul_oracle.h"
//#include "sqldb_sul_regex_oracle.h"
//#include "sqldb_sul_random_oracle.h"

#include <stdexcept>

using namespace std;

/**
 * @brief Does what you think it does. Since not every oracle requires access to distinguishing sequences, but when it does it normally shares them with 
 * other recourses like the algorithms, we require a shared pointer here, but allow it to be nullptr-initialized.
 */
unique_ptr<base_oracle> oracle_factory::create_oracle(const shared_ptr<sul_base>& sul, string_view oracle_name, const shared_ptr<distinguishing_sequences_handler_base>& ds_handler){
  if(AL_ORACLE == "base_oracle")
      return make_unique<base_oracle>(sul);
  
  if(AL_ORACLE == "input_file_oracle")
      return make_unique<input_file_oracle>(sul);
  else if(AL_ORACLE == "paul_oracle")
      return make_unique<paul_oracle>(sul, ds_handler);
  else if(AL_ORACLE == "sqldb_sul_random_oracle")
      throw std::runtime_error("Invalid oracle: Not included at the moment");
      //return make_unique<sqldb_sul_random_oracle>(sul);
  else if(AL_ORACLE == "sqldb_sul_regex_oracle")
      throw std::logic_error("Not implemented yet");
      //return make_unique<sqldb_sul_regex_oracle>(sul);
  else
    throw std::invalid_argument("One of the oracle specifying input parameters was not recognized by the oracle factory.");
}