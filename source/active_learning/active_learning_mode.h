/**
 * @file active_learning_mode.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The class maintaining the main subroutine that is starting the active learning.
 * @version 0.1
 * @date 2023-03-08
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _ACTIVE_LEARNING_MAIN_H_
#define _ACTIVE_LEARNING_MAIN_H_

#include "running_mode_base.h"

class active_learning_mode : public running_mode_base {
  public:
    void initialize() override;
    int run() override;
    void generate_output() override;
};

#endif
