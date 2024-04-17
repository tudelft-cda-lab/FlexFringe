/**
 * @file lsharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The (strategic) L#-algorithm, as described by Vandraager et al. (2022): "A New Approach for Active Automata
 * Learning Based on Apartness"
 * @version 0.1
 * @date 2023-03-11
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "lsharp.h"
#include "base_teacher.h"
#include "common_functions.h"
#include "input_file_oracle.h"
#include "input_file_sul.h"

#include "greedy.h"
#include "inputdata.h"
#include "main_helpers.h"
#include "mem_store.h"
#include "parameters.h"
#include "state_merger.h"

#include "edsm.h"

#include <list>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief For inheritance reasons. Saves us a second proc_counterexample implementation in derived classes.
 */
void lsharp_algorithm::update_state(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                                    const std::vector<int>& alphabet) const {
    return;
}

/**
 * @brief Processing the counterexample recursively in the binary search strategy
 * as described by the paper.
 *
 * Operations done directly on the APTA.
 *
 * @param aut The merged APTA.
 * @param counterex The counterexample.
 */
void lsharp_algorithm::proc_counterex(const unique_ptr<base_teacher>& teacher, inputdata& id,
                                      unique_ptr<apta>& hypothesis, const vector<int>& counterex,
                                      unique_ptr<state_merger>& merger, const refinement_list refs,
                                      const vector<int>& alphabet) const {
    // in this block we do a linear search for the fringe of the prefix tree. Once we found it, we ask membership
    // queries for each substring of the counterexample (for each new state that we create), and this way add the whole
    // counterexample to the prefix tree
    reset_apta(merger.get(), refs);
    vector<int> substring;
    apta_node* n = hypothesis->get_root();
    for (auto s : counterex) {
        if (n == nullptr) {
            const auto queried_type = teacher->ask_membership_query(substring, id);
            trace* new_trace = vector_to_trace(substring, id, queried_type);
            id.add_trace_to_apta(new_trace, hypothesis.get(), false);
            id.add_trace(new_trace);
            substring.push_back(s);
        } else {
            // find the fringe
            substring.push_back(s);
            trace* parse_trace = vector_to_trace(substring, id, 0); // TODO: inefficient like this
            tail* t = substring.size() == 0 ? parse_trace->get_end() : parse_trace->get_end()->past_tail;
            n = active_learning_namespace::get_child_node(n, t);
            mem_store::delete_trace(parse_trace);
        }
    }

    // for the last element, too
    const auto queried_type = teacher->ask_membership_query(substring, id);
    trace* new_trace = vector_to_trace(substring, id, queried_type);
    id.add_trace_to_apta(new_trace, hypothesis.get(), false);
    id.add_trace(new_trace);

    // now let's walk over the apta again, completing all the states we created
    n = hypothesis->get_root();
    trace* parsing_trace = vector_to_trace(counterex, id);
    tail* t = parsing_trace->get_head();
    while (n != nullptr) {
        update_state(merger, n, id, alphabet);
        n = active_learning_namespace::get_child_node(n, t);
        t = t->future();
    }
}

/**
 * @brief Take a node and complete it wrt to the alphabet.
 *
 * This is the only function that creates new nodes (apart from the root node).
 *
 */
unordered_set<apta_node*> lsharp_algorithm::extend_fringe(unique_ptr<state_merger>& merger, apta_node* n,
                                                                   unique_ptr<apta>& the_apta, inputdata& id,
                                                                   const vector<int>& alphabet) const {
    static unordered_set<apta_node*> extended_nodes; // TODO: do we need this data structure here?
    if (extended_nodes.contains(n))
        return unordered_set<apta_node*>();

    unordered_set<apta_node*> new_nodes;
    auto access_trace = n->get_access_trace();

    pref_suf_t seq;
    if (n->get_number() != -1 && n->get_number() != 0)
        seq = access_trace->get_input_sequence(true, true);
    else
        seq.resize(1);

    for (const int symbol : alphabet) {
        seq[seq.size() - 1] = symbol;

        const auto answer = teacher->ask_membership_query(seq, id);
        if(answer==-1)
            continue;

        trace* new_trace = vector_to_trace(seq, id, answer);
        id.add_trace_to_apta(new_trace, merger->get_aut(), false);
        new_nodes.insert(n->get_child(symbol));
    }
    return new_nodes;
}

/**
 * @brief Does what you think it does.
 *
 * @param merger
 * @param the_apta
 * @return list<refinement*>
 */
list<refinement*> lsharp_algorithm::find_complete_base(unique_ptr<state_merger>& merger,
                                                                unique_ptr<apta>& the_apta, inputdata& id,
                                                                const vector<int>& alphabet) {
    static int depth = 1; // because we initialize root node with depth 1
    static const int MAX_DEPTH = 29;
    static const bool COUNTEREXAMPLE_STRATEGY = false;

    list<refinement*> performed_refs;
    unordered_set<apta_node*> fringe_nodes; // TODO: it is wiser to put this into the extend_fringe method as static object

    int merge_depth = 0;

    while (true) { // cancel when either not red node identified or max depth

        unordered_set<apta_node*> blue_nodes;
        for (blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it) {
            auto blue_node = *b_it;
            blue_nodes.insert(blue_node);
        }

        bool reached_fringe = blue_nodes.empty();
        if (reached_fringe) {
            cout << "Reached fringe. Extend and recompute merges" << endl;
            reset_apta(merger.get(), performed_refs); // TODO: do we need to reset here?
            performed_refs.clear();

            for (auto fringe_node : fringe_nodes) {
                auto cn = extend_fringe(merger, fringe_node, the_apta, id, alphabet);
            }
            fringe_nodes.clear();

            merge_depth = 0;
            ++depth;
            continue;
        } else if (depth == MAX_DEPTH && COUNTEREXAMPLE_STRATEGY){
            static const int MAX_CEX_ITER = 20;
            static int c_iter = 0;
            cout << "Continuing search using the counterexample strategy. Iteration " << c_iter << " out of " << MAX_CEX_ITER << endl;
            find_closed_automaton(performed_refs, the_apta, merger, evidence_driven::get_score);
            c_iter++;
            if(c_iter==MAX_CEX_ITER){
                cout << "Max number of counterexample iterations reached. Printing automaton." << endl;
                print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
                exit(0);
            }
            return performed_refs;
        } else if (depth == MAX_DEPTH) {
            cout << "Max-depth reached and counterexample strategy disabled. Printing the automaton. Depth:" << depth << endl;
            find_closed_automaton(performed_refs, the_apta, merger, evidence_driven::get_score);
            print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
            cout << "Printed. Terminating" << endl;
            exit(0);
        }

        // go through each newly found fringe node, see if you can merge or extend
        ++merge_depth;
        if (merge_depth <= depth)
            fringe_nodes.clear();
        bool identified_red_node = false;
        //cout << "Looking for refinements" << endl;
        for (auto blue_node : blue_nodes) {
            if (merge_depth <= depth)
                fringe_nodes.insert(blue_node);

            refinement_set possible_merges;
            for (red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it) {
                const auto red_node = *r_it;

                refinement* ref = merger->test_merge(red_node, blue_node);
                if (ref != nullptr) {
                    possible_merges.insert(ref);
                }
            }

            if (possible_merges.size() == 0) {
                identified_red_node = true;
                refinement* ref = mem_store::create_extend_refinement(merger.get(), blue_node);
                ref->doref(merger.get());
                performed_refs.push_back(ref);
            } else {
                // get the best refinement from the heap
                refinement* best_merge = *(possible_merges.begin());
                for (auto it : possible_merges) {
                    if (it != best_merge)
                        it->erase();
                }
                best_merge->doref(merger.get());
                performed_refs.push_back(best_merge);
            }
        }

        //{
        //    static int model_nr = 0;
        //    cout << "printing model " << model_nr << " at depth " << depth << endl;
        //    print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_refs");
        //}

        if (!identified_red_node) {
            //cout << "Complete basis found. Forwarding hypothesis" << endl;
            find_closed_automaton(performed_refs, the_apta, merger, evidence_driven::get_score);
            return performed_refs;
        } else if (!reached_fringe) {
            //cout << "Processed layer " << merge_depth << endl;
            continue;
        }
    }
}

void lsharp_algorithm::run(inputdata& id) {
    int n_runs = 1;

    auto eval = unique_ptr<evaluation_function>(get_evaluation());
    eval->initialize_before_adding_traces();

    auto the_apta = unique_ptr<apta>(new apta());
    auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));

    const vector<int> alphabet = id.get_alphabet();
    cout << "Alphabet: ";
    active_learning_namespace::print_sequence<vector<int>::const_iterator>(alphabet.cbegin(), alphabet.cend());

    {
        // init the root node, s.t. we have blue states to iterate over
        auto root_node = the_apta->get_root();
        pref_suf_t seq;
        extend_fringe(merger, root_node, the_apta, id, alphabet);
    }

    //{
    //    static int model_nr = 0;
    //    print_current_automaton(merger.get(), "model.", "root");
    //}

   while (ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS) {
        if (n_runs % 100 == 0)
            cout << "Iteration " << n_runs + 1 << endl;

        auto refs = find_complete_base(merger, the_apta, id, alphabet);
        cout << "Searching for counterexamples" << endl;

        {
            static int model_nr = 0;
            cout << "printing model " << model_nr << endl;
            print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".before_cex");
        }

        // only merges performed, hence we can test our hypothesis
        while (true) {
            /* While loop to check the type. type is < 0 if sul cannot properly respond to query, e.g. when the string
            we ask cannot be parsed in automaton. We ignore those cases, as they lead to extra states in hypothesis.
            This puts a burden on the equivalence oracle to make sure no query is asked twice, else we end
            up in infinite loop.*/

            optional<pair<vector<int>, int>> query_result = oracle->equivalence_query(merger.get(), teacher);
            if (!query_result) {
                cout << "Found consistent automaton => Print." << endl;
                print_current_automaton(merger.get(), OUTPUT_FILE, ".final"); // printing the final model each time
                return;
            }

            const int type = query_result.value().second;
            if (type < 0)
                continue;

            const vector<int>& cex = query_result.value().first;
            cout << "Counterexample of length " << cex.size() << " found: ";
            print_vector(cex);
            proc_counterex(teacher, id, the_apta, cex, merger, refs, alphabet);

            break;
        }

        // TODO: this one might not hold
        //++n_runs;
        //if (ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS) {
        //    cout << "Maximum of runs reached. Printing automaton." << endl;
        //    print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
        //    return;
        //}
    }
}
