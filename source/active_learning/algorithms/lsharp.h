/**
 * @file lsharp.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The (strategic) L#-algorithm, as described by Vandraager et al. (2022): "A New Approach for Active Automata
 * Learning Based on Apartness"
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _L_SHARP_H_
#define _L_SHARP_H_

#include "algorithm_base.h"
#include "base_teacher.h"
#include "definitions.h"
#include "eq_oracle_base.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"

#include <list>
#include <memory>

class lsharp_algorithm : public algorithm_base {
  protected:
    //unordered_set< vector<int> > distinguishing_sequences;

    virtual void proc_counterex(const std::unique_ptr<base_teacher>& teacher, inputdata& id, unique_ptr<apta>& hypothesis,
                        const std::vector<int>& counterex, std::unique_ptr<state_merger>& merger,
                        const refinement_list refs, const vector<int>& alphabet) const;

    virtual void extend_fringe(std::unique_ptr<state_merger>& merger, apta_node* n,
                                                 std::unique_ptr<apta>& the_apta, inputdata& id,
                                                 const vector<int>& alphabet) const;

    void update_state(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                      const std::vector<int>& alphabet) const;

    virtual list<refinement*> find_complete_base(unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta, inputdata& id,
                                         const std::vector<int>& alphabet);

  public:
    lsharp_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                     std::unique_ptr<eq_oracle_base>& oracle)
        : algorithm_base(sul, teacher, oracle){};

    virtual void run(inputdata& id) override;
};

#endif
