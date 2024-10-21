/**
 * @file probabilistic_lsharp_v2.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-10-10
 *
 * @copyright Copyright (c) 2023
 * 
 * This version is the same, but I implemented Sicco's recommendation. Might not matter in the end.
 *
 */

#ifndef _PROBABILISTIC_L_SHARP_V2_H_
#define _PROBABILISTIC_L_SHARP_V2_H_

#include "algorithm_base.h"
#include "lsharp.h"

#include "base_teacher.h"
#include "definitions.h"
#include "eq_oracle_base.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "string_probability_oracle.h"
#include "tail.h"
#include "trace.h"

#include <list>
#include <memory>
#include <unordered_map>

class probabilistic_lsharp_v2_algorithm : public lsharp_algorithm {
  private:
    bool MAX_DEPTH_REACHED = false;
    // bool underestimated_dist = false; // helper flag. If we see final prob larger than 1 in one node, we continue
    // extending the fringe.

    __attribute__((always_inline)) inline /* bool */ void update_tree_dfs(apta* the_apta,
                                                                          const std::vector<int>& alphabet) const;
    __attribute__((always_inline)) inline void update_access_path(apta_node* n, apta* the_apta,
                                                                  const std::vector<int>& alphabet) const;

    inline void add_statistics(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                               const std::vector<int>& alphabet,
                               std::optional<active_learning_namespace::pref_suf_t> seq_opt) const;

    __attribute__((always_inline)) inline std::unordered_set<apta_node*>
    extend_fringe(std::unique_ptr<state_merger>& merger, apta_node* n, std::unique_ptr<apta>& the_apta, inputdata& id,
                  const std::vector<int>& alphabet) const;

    inline void init_final_prob(apta_node* n, apta* the_apta, inputdata& id) const;
    std::list<refinement*> find_complete_base(std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta, inputdata& id,
                                         const std::vector<int>& alphabet);

  protected:
    inline void proc_counterex(const std::unique_ptr<base_teacher>& teacher, inputdata& id,
                               std::unique_ptr<apta>& hypothesis, const std::vector<int>& counterex,
                               std::unique_ptr<state_merger>& merger, const refinement_list refs,
                               const std::vector<int>& alphabet) const;

  public:
    probabilistic_lsharp_v2_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                                   std::unique_ptr<eq_oracle_base>& oracle)
        : lsharp_algorithm(sul, teacher, oracle) {
        std::cout << "Probabilistic L# only works with probabilistic oracle. Automatically switched to that one.\
If this is undesired behavior check your input and/or source code."
                  << std::endl;
        this->oracle.reset(new string_probability_oracle(sul));
    };

    virtual void run(inputdata& id) override;
};

#endif
