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
#include "input/trace.h"
#include "inputdata.h"
#include "refinement.h"
#include "sqldb_sul.h"
#include "sqldb_sul_random_oracle.h"
#include "sqldb_sul_regex_oracle.h"
#include "state_merger.h"
#include <memory>
#include <set>
#include <vector>

class ldot_algorithm : public algorithm_base {
  private:
    // START -- SETTINGS ******:

    int PREFIX_SIZE = 10;
    double BEST_MERGE_THRESHOLD = 2.0; // any value below 1 is a JustTakeBestMerge approach

    // Keep false, does not work very well, allthough Lsharp works with true.
    bool BLUE_NODE_COMPLETION = false;

    // Perform a specific query that tests if two nodes are equivalent
    bool DISTINGUISHING_MERGE_TEST = false;

    // When a cex is found:
    bool COMPLETE_PATH_CEX = true;   // This completes everything that can be known on the path.
    bool EXPLORE_OUTSIDE_CEX = true; // This uses a prefix query to explore the PREFIX_SIZE depth from its path.

    // EQUIVALENCE SETTINGS
    // If regex equivalence is first, that one starts
    // Then one of random or distinguishing is started.
    // Random takes precedence
    bool REGEX_EQUIVALENCE = false;
    bool RANDOM_EQUIVALENCE = true;
    bool DISTINGUISHING_EQUIVALENCE = false;

    // Also complete the represented nodes instead of only the representative.
    bool COMPLETE_REPRESENTED = false;

    // END -- SETTINGS ******

    /** Some object parameters */
    bool disable_regex_oracle = false;
    int n_runs;
    int n_subs;
    int uid = 0;
    std::list<refinement*> performed_refinements;
    unordered_set<apta_node*> completed_nodes;
    unordered_set<int> added_traces;
    bool isolated_states;
    std::set<apta_node*> complete_these_states;

    /** External structures */
    unique_ptr<evaluation_function> my_eval;
    unique_ptr<apta> my_apta;
    unique_ptr<state_merger> my_merger;
    shared_ptr<sqldb_sul> my_sul;
    unique_ptr<sqldb_sul_random_oracle> random_oracle;

    /**
     * Add a node to the list for nodes that should be completed.
     * Keeps track of nodes that are already completed
     */
    bool maybe_list_for_completion(apta_node* n);

    /**
     * For a specific node (n) "complete" the node.
     * That is, call a prefix query on the node,
     * thus exploring its outgoing branches and more.
     * For Lsharp, this was only all possible 1 steps.
     * Returns false is it is already completed or if no new information is added.
     */
    bool complete_state(inputdata& id, apta_node* n);

    /** Debugging function */
    void test_access_traces();

    /**
     * @brief Add a new trace to the data structures (apta, mem_store).
     * Returns true if trace has been added.
     */
    bool add_trace(inputdata& id, const psql::record& rec);

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
    void proc_counter_record(inputdata& id, const psql::record& rec, const refinement_list& refs);

    /** Function to process a std::vector with refinement_set (merges)
     * we are unsure of
     */
    void process_unidentified(inputdata& id, const std::vector<refinement_set>& refs_for_unidentified);

    /** Do a ref,
     * add it to the performed_refinements.
     */
    void merge_processed_ref(refinement* ref);

    /** Perform an equivalence check
     * Return an optional record holding the counter example
     */
    optional<psql::record> equivalence(sqldb_sul_regex_oracle* regex_oracle);

    std::vector<apta_node*> represented_by(apta_node* n);

  public:
    ldot_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                   std::unique_ptr<eq_oracle_base>& oracle)
        : algorithm_base(sul, teacher, oracle){};

    /**
     * @brief Call to run algorithm.
     */
    void run(inputdata& id) override;
};

#endif
