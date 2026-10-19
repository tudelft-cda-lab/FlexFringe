/**
 * @file ds_handler_factory.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-05-10
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "ds_handler_factory.h"

#include "distinguishing_sequences_handler.h"
#include "distinguishing_sequences_handler_fast.h"

#include <stdexcept>

using namespace std;

/**
 * @brief Creates the ds-handler.
 */
shared_ptr<distinguishing_sequences_handler_base> ds_handler_factory::create_ds_handler(const shared_ptr<sul_base>& sul, string_view handler_name){
  if(handler_name.size() == 0)
    return shared_ptr<distinguishing_sequences_handler_base>(nullptr);
  else if(handler_name == "distinguishing_sequences")
    return make_shared<distinguishing_sequences_handler>(sul);
  else if(handler_name == "distinguishing_sequences_fast")
    return make_shared<distinguishing_sequences_handler_fast>(sul);
  else
    throw logic_error("Invalid ii_handler name");
}