/**
 * @file ldot.h
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief The Ldot-algorithm, as part of my thesis
 * @version 0.1
 * @date 2024-1-9
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef _L_DOT_H_
#define _L_DOT_H_

#include "algorithm_base.h"
#include "apta.h"
#include "base_teacher.h"
#include "eq_oracle_base.h"
#include "inputdata.h"
#include "sqldb_sul.h"
#include "state_merger.h"
#include <memory>
#include <vector>

class ldot_algorithm : public algorithm_base {
  private:
    /**
     * For a specific node (n) complete for as possible all the 1 size steps.
     */
    void complete_state(inputdata& id, const std::vector<int>& alphabet, apta_node* n);

    /**
     * @brief Add a new trace to the data structures (apta, mem_store).
     */
    void add_trace(inputdata& id, std::vector<int> seq, int answer);

    /**
     * @brief Processing the counterexample recursively in the binary search strategy
     * as described by the paper.
     *
     * Operations done directly on the APTA.
     *
     * In this block we do a linear search for the fringe of the prefix tree. Once we found it, we ask membership
     * queries for each substring of the counterexample (for each new state that we create), and this way add the whole
     * counterexample to the prefix tree
     * @param id Inputdata object
     * @param alphabet The alphabet
     * @param counterex The counterexample.
     * @param refs The list of refinments made.
     */
    void proc_counterex(inputdata& id, const vector<int>& alphabet, const vector<int>& counterex,
                        const refinement_list& refs);

    std::vector<refinement*> process_unidentified(std::vector<refinement_set> refs_for_unidentified);
    unique_ptr<evaluation_function> my_eval;
    unique_ptr<apta> my_apta;
    unique_ptr<state_merger> my_merger;
    shared_ptr<sqldb_sul> my_sul;

  public:
    ldot_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                   std::unique_ptr<eq_oracle_base>& oracle)
        : algorithm_base(sul, teacher, oracle){};

    void run(inputdata& id) override;
};

#endif
