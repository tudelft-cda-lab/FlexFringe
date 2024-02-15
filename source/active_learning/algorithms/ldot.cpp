/**
 * @file ldot.h
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief The Ldot-algorithm, as part of my thesis
 * @version 0.1
 * @date 2024-1-9
 *
 * @copyright Copyright (c) 2024
 *
 * This code is heavily inspired (and copied) by the lsharp.cpp implementation by Robert Baumgartner.
 */

#include "ldot.h"
#include "common_functions.h"
#include "evaluate.h"
#include "loguru.hpp"
#include "main_helpers.h"
#include "misc/printutil.h"
#include "parameters.h"
#include "refinement.h"
#include "sqldb_sul.h"
#include <fmt/format.h>
#include <memory>
#include <vector>

void ldot_algorithm::proc_counterex(inputdata& id, const vector<int>& alphabet, const vector<int>& counterex,
                                    const refinement_list& refs) {
    active_learning_namespace::reset_apta(my_merger.get(), refs);
    vector<int> substring;
    apta_node* n = my_apta->get_root();
    for (auto s : counterex) {
        if (n == nullptr) {
            const int queried_type = my_sul->query_trace_maybe(substring);
            // TODO: How to deal with gaps. ie queried_type is -1.
            trace* new_trace = active_learning_namespace::vector_to_trace(substring, id, queried_type);
            id.add_trace_to_apta(new_trace, my_apta.get(), false);
            id.add_trace(new_trace);
            substring.push_back(s);
        } else {
            // find the fringe
            substring.push_back(s);
            trace* parse_trace =
                active_learning_namespace::vector_to_trace(substring, id, 0); // TODO: inefficient like this
            tail* t = substring.size() == 0 ? parse_trace->get_end() : parse_trace->get_end()->past_tail;
            n = active_learning_namespace::get_child_node(n, t);
            mem_store::delete_trace(parse_trace);
        }
    }

    // for the last element, too
    const int queried_type = my_sul->query_trace_maybe(substring);
    // TODO: How to deal with gaps. ie queried_type is -1.
    trace* new_trace = active_learning_namespace::vector_to_trace(substring, id, queried_type);
    id.add_trace_to_apta(new_trace, my_apta.get(), false);
    id.add_trace(new_trace);

    // now let's walk over the apta again, completing all the states we created
    n = my_apta->get_root();
    trace* parsing_trace = active_learning_namespace::vector_to_trace(counterex, id);
    tail* t = parsing_trace->get_head();
    while (n != nullptr) {
        complete_state(id, alphabet, n);
        n = active_learning_namespace::get_child_node(n, t);
        t = t->future();
    }
}
void ldot_algorithm::add_trace(inputdata& id, std::vector<int> seq, int answer) {
    trace* new_trace = active_learning_namespace::vector_to_trace(seq, id, answer);
    id.add_trace_to_apta(new_trace, my_merger->get_aut(), false);
    id.add_trace(new_trace);
}

std::vector<refinement*> ldot_algorithm::process_unidentified(std::vector<refinement_set> refs_for_unidentified) {
    // SOME HEURISTIC THAT BALANCES EXPLORATION VS EXPLOITATION.
    // What to do when a state is unidentified (subroutine LdotProcessUnidentified). Here we take some
    // different approaches and we should see what works best. Additionally, what is in the unidentified
    // portion might prove that more exploration is needed, instead of building the hypothesis
    // (exploitation part).
    // Perhaps some weak co-transitivity test can also be done here.
    // If an unidentified state has a lot of high merge probabilities we need to find more information about
    // these states.
    // Keep track of the explored depth of a state and if a state has a very low explored depth, we need
    // to increase the depth that is explored for that state.
    // For unidentified states, we might already have a very high merge probability. We can then make this
    // merge.

    std::vector<refinement*> selected_refs;

    for (auto possible_refs : refs_for_unidentified) {
        refinement* best_merge = *possible_refs.begin(); // TODO: is this best merge?
        selected_refs.push_back(best_merge);
    }
    return selected_refs;
}

void ldot_algorithm::complete_state(inputdata& id, const vector<int>& alphabet, apta_node* n) {
    static unordered_set<apta_node*> completed_nodes;
    if (completed_nodes.contains(n))
        return;

    // TODO: Optimize with a prefix query.
    for (const int symbol : alphabet) {
        if (n->get_child(symbol) == nullptr) {
            auto* access_trace = n->get_access_trace();

            active_learning_namespace::pref_suf_t seq = {symbol};
            if (n->get_number() != -1) {
                seq = access_trace->get_input_sequence(true, true);
                seq[seq.size() - 1] = symbol;
            }

            const int answer = my_sul->query_trace_maybe(seq);
            if (answer != -1)
                add_trace(id, seq, answer);
        }
    }
    completed_nodes.insert(n);
}

void ldot_algorithm::run(inputdata& id) {
    LOG_S(INFO) << "Running the ldot algorithm.";
    my_sul = dynamic_pointer_cast<sqldb_sul>(sul);
    if (my_sul == nullptr) {
        throw logic_error("ldot only works with sqldb_sul.");
    }
    int n_runs = 1;

    // Initialize data structures.
    my_eval = unique_ptr<evaluation_function>(get_evaluation());
    my_eval->initialize_before_adding_traces();
    my_apta = make_unique<apta>();
    my_merger = make_unique<state_merger>(&id, my_eval.get(), my_apta.get());
    const vector<int> my_alphabet = id.get_alphabet();

    // init the root node, s.t. we have blue states to iterate over
    complete_state(id, my_alphabet, my_apta->get_root());
    print_current_automaton(my_merger.get(), "debug", "");

    while (ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS) {
        if (n_runs % 100 == 0)
            LOG_S(INFO) << "Iteration " << n_runs + 1;

        bool no_isolated_states = true; // avoid iterating over a changed data structure (apta)
        std::list<refinement*> performed_refinements;
        // The refinements for the unidentified nodes.
        std::vector<refinement_set> refs_for_unidentified;

        for (blue_state_iterator b_it = blue_state_iterator(my_apta->get_root()); *b_it != nullptr; ++b_it) {
            const auto blue_node = *b_it;

            if (blue_node->get_size() == 0)
                throw logic_error("This should never happen: TODO: Delete line");

            // This is a difference with Vandraager (they only complete red nodes),
            // but we it improved statistical methods
            // TODO: Robert: do we need this one here really? I can't see it at the moment why
            // TODO: Hielke: For ldot just make an improved prefix query as initialize more states.
            complete_state(id, my_alphabet, blue_node);

            refinement_set possible_refs;
            for (red_state_iterator r_it = red_state_iterator(my_apta->get_root()); *r_it != nullptr; ++r_it) {
                auto* const red_node = *r_it;

                refinement* ref = my_merger->test_merge(red_node, blue_node);
                if (ref != nullptr) {
                    // Collect all possible refinements.
                    possible_refs.insert(ref);
                }
            }

            if (possible_refs.size() == 0) {
                // Promotion of the blue node.
                no_isolated_states = false;
                refinement* ref = mem_store::create_extend_refinement(my_merger.get(), blue_node);
                ref->doref(my_merger.get());
                performed_refinements.push_back(ref);
            }

            if (possible_refs.size() == 1) {
                // There is one refinement, do that one.
                refinement* ref = *possible_refs.begin();
                ref->doref(my_merger.get());
                performed_refinements.push_back(ref);
            }

            if (possible_refs.size() > 1) {
                // Check these possible refinements later to perform or to explore more.
                refs_for_unidentified.push_back(possible_refs);
            }
        }

        if (!refs_for_unidentified.empty()) {
            // Select based on some criteria some refs
            // Also might make some queries to get additional info for these unidentified blue nodes.
            auto selected_refs = process_unidentified(refs_for_unidentified);
            for (auto* ref : selected_refs) {
                ref->doref(my_merger.get());
                performed_refinements.push_back(ref);
            }
        }

        if (!no_isolated_states) {
            // There are still isolated states that need to be resolved.
            continue;
        }

        // build hypothesis
        active_learning_namespace::minimize_apta(performed_refinements, my_merger.get());

        static int model_nr = 0;
        print_current_automaton(my_merger.get(), "model.",
                                to_string(++model_nr) + ".after_ref"); // printing the final model each time

        // Check equivalence, if not process the counter example.
        while (true) {
            /* While loop to check the type. type is < 0 if sul cannot properly respond to query, e.g. when the
            string we ask cannot be parsed in automaton. We ignore those cases, as they lead to extra states in
            hypothesis. This puts a burden on the equivalence oracle to make sure no query is asked twice, else we
            end up in infinite loop.*/

            optional<pair<vector<int>, int>> query_result = oracle->equivalence_query(my_merger.get(), teacher);
            if (!query_result) {
                LOG_S(INFO) << "Found consistent automaton => Print.";
                print_current_automaton(my_merger.get(), OUTPUT_FILE, ".final"); // printing the final model each time
                return;                                                          // Solved!
            }

            const int type = query_result.value().second;
            // If this query could not be found ask for a new counter example.
            if (type < 0)
                continue;

            const vector<int>& cex = query_result.value().first;
            std::stringstream ss;
            ss << cex;
            LOG_S(INFO) << "Counterexample of length " << cex.size() << " found: " << ss.str();
            proc_counterex(id, my_alphabet, cex, performed_refinements);
            break;
        }

        ++n_runs;
        if (ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS) {
            LOG_S(INFO) << "Maximum of runs reached. Printing automaton.";
            for (auto* top_ref : performed_refinements) { top_ref->doref(my_merger.get()); }
            print_current_automaton(my_merger.get(), OUTPUT_FILE, ".final");
            return;
        }
    }
}
