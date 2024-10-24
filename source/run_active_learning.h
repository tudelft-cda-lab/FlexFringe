/**
 * @file run_active_learning.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Header for the active learning main routine.
 * @version 0.1
 * @date 2023-02-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _ACTIVE_LEARNING_H_
#define _ACTIVE_LEARNING_H_

#include "state_merger.h"
#include "refinement.h"

void run_active_learning(state_merger* merger);

#endif /* _ACTIVE_LEARNING_H_ */
