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
#include "parameters.h"
#include "lstar.h"

#include <stdexcept>

using namespace std;
using namespace active_learning_namespace;

void active_learning_namespace::run_active_learning(){
  if(ACTIVE_LEARNING_ALGORITHM == "l_star"){
    auto l_star = lstar_algorithm();
  }
  else{
    throw logic_error("Fatal error: Unknown active_learning_algorithm flag used: " + ACTIVE_LEARNING_ALGORITHM);
  }
}