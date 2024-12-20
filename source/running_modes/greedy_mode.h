/**
 * @file greedy_mode.h
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _RANDOM_GREEDY_H_
#define _RANDOM_GREEDY_H_

#include "running_mode_base.h"
#include "state_merger.h"
#include "refinement.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>

/**
 * @brief The classical greedy run mode: Contruct and APTA, then minimize via state merging.
 * 
 */
class greedy_mode : public running_mode_base {
  private:
    const int RANDOMG = 1;
    const int NORMALG = 2;

  public:
    void initialize() override;
    int run() override;
    void generate_output() override;
};

#endif /* _RANDOM_GREEDY_H_ */
