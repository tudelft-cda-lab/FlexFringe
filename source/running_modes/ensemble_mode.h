/**
 * @file ensemble_mode.h
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _ENSEMBLE_MODE_H_
#define _ENSEMBLE_MODE_H_

#include "running_mode_base.h"
#include "state_merger.h"
#include "refinement.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>

class ensemble_mode : public running_mode_base {
  protected:
    refinement_list* greedy();
    void bagging(std::string output_file, int nr_estimators);

  public:
    int run() override;
    void initialize() override;
};

#endif // _ENSEMBLE_MODE_H_
