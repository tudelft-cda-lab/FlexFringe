/**
 * @file active_learning_main.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The main subroutine that is starting the active learning.
 * @version 0.1
 * @date 2023-03-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _ACTIVE_LEARNING_MAIN_H_
#define _ACTIVE_LEARNING_MAIN_H_

#include <vector>

namespace active_learning_namespace{
  const std::vector<int> get_alphabet();

  void run_active_learning();
}

#endif