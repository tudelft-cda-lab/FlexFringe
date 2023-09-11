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

#ifndef _PROBABILISTIC_L_SHARP_H_
#define _PROBABILISTIC_L_SHARP_H_

#include "algorithm_base.h"
#include "lsharp.h"

#include "state_merger.h"
#include "inputdata.h"
#include "definitions.h"
#include "trace.h"
#include "tail.h"
#include "refinement.h"
#include "base_teacher.h"
#include "eq_oracle_base.h"
#include "probabilistic_oracle.h"

#include <list> 
#include <memory>
#include <unordered_map>

class probabilistic_lsharp_algorithm : public lsharp_algorithm {
  protected:

    
    virtual void complete_state(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const std::vector<int>& alphabet) const override;
    virtual void update_state(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const std::vector<int>& alphabet) const override;

  public:
    probabilistic_lsharp_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher, std::unique_ptr<eq_oracle_base>& oracle) 
      : lsharp_algorithm(sul, teacher, oracle){
        std::cout << "Probabilistic L# only works with probabilistic oracle. Automatically switched to that one.\
        If this is undesired behavior check your input and/or source code." << std::endl;
        this->oracle.reset(new probabilistic_oracle(sul));
      };
};

#endif
