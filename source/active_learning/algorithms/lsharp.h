/**
 * @file lsharp.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The (strategic) L#-algorithm, as described by Vandraager et al. (2022): "A New Approach for Active Automata Learning Based on Apartness"
 * @version 0.1
 * @date 2023-02-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _L_SHARP_H_
#define _L_SHARP_H_

#include "state_merger.h"
#include "inputdata.h"
#include "definitions.h"
#include "trace.h"
#include "tail.h"
#include "refinement.h"

#include <vector> 
#include <memory>

class lsharp_algorithm{
  protected:
    const std::vector<int> alphabet;
    vector< refinement* > construct_automaton_from_table(std::unique_ptr<state_merger>& merger, inputdata& id) const;

    void init_root_state(apta* aut) const;
    void proc_counterex(apta* aut, const std::vector<int>& counterex) const;

    

  public:
    lsharp_algorithm(const std::vector<int>& alphabet);
    void run_l_sharp();
};

#endif
