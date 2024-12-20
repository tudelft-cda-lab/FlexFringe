/**
 * @file interactive_mode.h
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _INTERACTIVE_MODE_H_
#define _INTERACTIVE_MODE_H_

#include "running_mode_base.h"
#include "state_merger.h"
#include "refinement.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>

class interactive_mode : public running_mode_base {
  public:
    int run() override;
    void initialize() override;
    void generate_output() override;
};

#endif /* _INTERACTIVE_MODE_H_ */
