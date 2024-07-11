/**
 * @file weighted_lsharp.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-11-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _TRANSFORMER_L_SHARP_H_
#define _TRANSFORMER_L_SHARP_H_

#include "algorithm_base.h"
#include "lsharp.h"

#include "base_teacher.h"
#include "definitions.h"
#include "active_state_sul_oracle.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"
#include "membership_state_comparator.h"

#include <list>
#include <memory>
#include <unordered_map>

class transformer_lsharp_algorithm : public lsharp_algorithm {
  private:
    const bool use_sinks;
  protected:
    void proc_counterex(const std::unique_ptr<base_teacher>& teacher, inputdata& id, unique_ptr<apta>& hypothesis,
                        const std::vector<int>& counterex, std::unique_ptr<state_merger>& merger,
                        const refinement_list refs, const vector<int>& alphabet) const;

    void extend_fringe(std::unique_ptr<state_merger>& merger, apta_node* n,
                                                 std::unique_ptr<apta>& the_apta, inputdata& id,
                                                 const vector<int>& alphabet) const;
    virtual void update_hidden_states(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                               const std::vector<int>& alphabet, const std::optional< std::vector< std::vector<float > > >& hidden_states,
                               std::optional<active_learning_namespace::pref_suf_t> seq_opt) const;

    list<refinement*> find_complete_base(unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta, inputdata& id,
                                         const std::vector<int>& alphabet) override;

  public:
    transformer_lsharp_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                              std::unique_ptr<eq_oracle_base>& oracle)
        : lsharp_algorithm(sul, teacher, oracle), use_sinks(USE_SINKS) {
        std::cout << "Probabilistic L# only works with probabilistic oracle. Automatically switched to that one.\
If this is undesired behavior check your input and/or source code."
                  << std::endl;
        this->oracle.reset(new active_state_sul_oracle(sul));
    };

    virtual void run(inputdata& id) override;
};

#endif
