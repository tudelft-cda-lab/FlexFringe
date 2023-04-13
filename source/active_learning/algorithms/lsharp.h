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

#include "algorithm_base.h"
#include "state_merger.h"
#include "inputdata.h"
#include "definitions.h"
#include "trace.h"
#include "tail.h"
#include "refinement.h"
#include "base_teacher.h"
#include "eq_oracle_base.h"

#include <vector> 
#include <memory>

class lsharp_algorithm : public algorithm_base {
  protected:
    vector< refinement* > construct_automaton_from_table(std::unique_ptr<state_merger>& merger, inputdata& id) const;

    void complete_state(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const vector<int>& alphabet) const;
    void proc_counterex(apta* aut, const std::vector<int>& counterex) const;
    refinement* extract_best_merge(refinement_set* rs) const;

  public:
    lsharp_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher, std::unique_ptr<eq_oracle_base>& oracle) 
      : algorithm_base(sul, teacher, oracle){};
    void run(inputdata& id) override;
};

#endif
