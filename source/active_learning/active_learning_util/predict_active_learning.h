/**
 * @file predict_active_learning.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-03-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _ACTIVE_LEARNING_PREDICT_
#define _ACTIVE_LEARNING_PREDICT_

#include "fstream"
#include "state_merger.h"
#include "inputdata.h"

#include <iostream>
//#include <cstdlib>

namespace active_learning_namespace{
  void predict(state_merger* m, inputdata& id, std::ofstream& output);
}

#endif