/**
 * @file lstar.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _L_STAR_H_
#define _L_STAR_H_

#include "observation_table.h"

#include <vector> 

class lstar_algorithm{
  protected:
    observation_table obs_table;

    void construct_automaton_from_table();
    
  public:
    lstar_algorithm(std::vector<int>& alphabet);
    void run_l_star();
};

#endif
