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
#include "state_merger.h"
#include "inputdata.h"
#include "definitions.h"
#include "trace.h"
#include "tail.h"
#include "refinement.h"

#include <vector> 
#include <memory>
#include <list>

class lstar_algorithm{
  protected:
    const std::list< refinement* > construct_automaton_from_table(const observation_table& obs_table, std::unique_ptr<state_merger>& merger, inputdata& id) const;
  public:
    lstar_algorithm() = default;
    void run_l_star(inputdata& id);
};

#endif
