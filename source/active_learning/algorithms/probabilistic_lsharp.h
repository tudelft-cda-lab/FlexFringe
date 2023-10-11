/**
 * @file probabilistic_lsharp.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-10-10
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
    inline void proc_counterex(const std::unique_ptr<base_teacher>& teacher, inputdata& id, unique_ptr<apta>& hypothesis, 
                        const std::vector<int>& counterex, std::unique_ptr<state_merger>& merger, const refinement_list refs,
                        const vector<int>& alphabet) const;

    __attribute__((always_inline)) inline bool extend_fringe(std::unique_ptr<state_merger>& merger, apta_node* n, std::unique_ptr<apta>& the_apta, 
                                                             inputdata& id, const vector<int>& alphabet) const;
    inline void add_statistics(std::unique_ptr<state_merger>& merger, apta_node* n,inputdata& id, 
                                                        const std::vector<int>& alphabet, std::optional< active_learning_namespace::pref_suf_t > seq_opt) const;
    
    __attribute__((always_inline)) inline void update_tree_recursively(apta_node* n, apta* the_apta, const std::vector<int>& alphabet) const;
    __attribute__((always_inline)) inline void update_tree_dfs(apta* the_apta, const std::vector<int>& alphabet) const;

    inline void update_final_probability(apta_node* n, apta* the_apta) const;
    inline void init_final_prob(apta_node* n, apta* the_apta, inputdata& id) const;

  public:
    probabilistic_lsharp_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher, std::unique_ptr<eq_oracle_base>& oracle) 
      : lsharp_algorithm(sul, teacher, oracle){
        std::cout << "Probabilistic L# only works with probabilistic oracle. Automatically switched to that one.\
        If this is undesired behavior check your input and/or source code." << std::endl;
        this->oracle.reset(new probabilistic_oracle(sul));
      };

    virtual void run(inputdata& id) override;
};

#endif
