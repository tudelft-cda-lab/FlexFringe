/**
 * @file lstar_imat.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _L_STAR_IMAT_H_
#define _L_STAR_IMAT_H_

#include "algorithm_base.h"
#include "definitions.h"
#include "inputdata.h"
#include "observation_table_imat.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"

#include <list>
#include <memory>
#include <set>

class lstar_imat_algorithm : public algorithm_base {
  private:
    const std::list<refinement*> construct_automaton_from_table(const observation_table_imat& obs_table,
                                                                std::unique_ptr<state_merger>& merger, inputdata& id);
    std::set<std::vector<int>> added_traces;

  public:
    lstar_imat_algorithm(std::unique_ptr<oracle_base>&& oracle)
        : algorithm_base(std::move(oracle)){};
    void run(inputdata& id) override;
};

#endif
