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
#include "input/trace.h"
#include "loguru.hpp"
#include "main_helpers.h"
#include "misc/printutil.h"
#include "parameters.h"
#include "refinement.h"
#include "sqldb_sul.h"
#include <fmt/format.h>
#include <memory>
#include <vector>

template <typename T> std::string printAddress(const std::list<T>& v) {
    stringstream ss;
    ss << "[";
    for (auto& e : v) ss << &e << ", ";
    ss << "]\n";
    return ss.str();
}

void ldot_algorithm::proc_counterex(inputdata& id, const vector<int>& alphabet, const vector<int>& counterex,
                                    const refinement_list& refs) {
    active_learning_namespace::reset_apta(my_merger.get(), refs);

#ifndef NDEBUG
    DLOG_S(1) << n_runs << ": Undoing " << refs.size() << " refs.";
    // printing the model each time
    print_current_automaton(my_merger.get(), "debug/model.", std::to_string(n_runs) + ".zzz");
#endif

    performed_refinements.clear();

    vector<int> substring;
    apta_node* n = my_apta->get_root();

    for (auto s : counterex) {
        substring.push_back(s);
        if (n != nullptr) {
            // Look for fringe, then add to apta.
            n = active_learning_namespace::get_child_node(n, s);
            continue;
        }
        const int queried_type = my_sul->query_trace_maybe(substring);
        if (queried_type != -1) {
            add_trace(id, substring, queried_type);
        }
    }

    // now let's walk over the apta again, completing all the states we created
    n = my_apta->get_root();
    for (auto s : counterex) {
        n = active_learning_namespace::get_child_node(n, s);
        complete_state(id, alphabet, n);
    }
}

trace* ldot_algorithm::add_trace(inputdata& id, const std::vector<int>& seq, int answer) {
    stringstream ss;
    ss << answer << " " << seq;
    DLOG_S(1) << ss.str();

    trace* new_trace = active_learning_namespace::vector_to_trace(seq, id, answer);
    id.add_trace_to_apta(new_trace, my_merger->get_aut(), false);
    id.add_trace(new_trace);
    return new_trace;
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
    n_runs = 1;

    // Initialize data structures.
    my_eval = unique_ptr<evaluation_function>(get_evaluation());
    my_eval->initialize_before_adding_traces();
    my_apta = make_unique<apta>();
    my_merger = make_unique<state_merger>(&id, my_eval.get(), my_apta.get());
    my_apta->set_context(my_merger.get());
    my_eval->set_context(my_merger.get());
    const vector<int> my_alphabet = id.get_alphabet();

    // init the root node, s.t. we have blue states to iterate over
    complete_state(id, my_alphabet, my_apta->get_root());
    print_current_automaton(my_merger.get(), "debug", "");

    while (ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS) {
        if (n_runs % 100 == 0)
            LOG_S(INFO) << "Iteration " << n_runs + 1;

        bool no_isolated_states = true; // avoid iterating over a changed data structure (apta)
        // The refinements for the unidentified nodes.
        std::vector<refinement_set> refs_for_unidentified;

#ifndef NDEBUG
        // printing the model each time (once per n_runs).
        static int x = -1;
        static int uid = 0;
        if (x != n_runs)
            print_current_automaton(my_merger.get(), "debug/model.", std::to_string(n_runs) + ".bef");
        x = n_runs;
#endif

        for (blue_state_iterator b_it = blue_state_iterator(my_apta->get_root()); *b_it != nullptr; ++b_it) {
            auto* blue_node = *b_it;

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

            if (possible_refs.empty()) {
                // Promotion of the blue node.
                no_isolated_states = false;
                refinement* ref = mem_store::create_extend_refinement(my_merger.get(), blue_node);

#ifndef NDEBUG
                print_current_automaton(my_merger.get(), "debug/model.",
                                        std::to_string(n_runs) + fmt::format(".f{:0>4}", uid++) + ".ext_a");
                ref->doref(my_merger.get());
                performed_refinements.push_back(ref);
                print_current_automaton(my_merger.get(), "debug/model.",
                                        std::to_string(n_runs) + fmt::format(".f{:0>4}", uid++) + ".ext_b");
                DLOG_S(2) << n_runs << ":extend:" << printAddress(performed_refinements);
#else
                ref->doref(my_merger.get());
                performed_refinements.push_back(ref);
#endif
            }

            if (possible_refs.size() == 1) {
                // There is one refinement, do that one.
                refinement* ref = *possible_refs.begin();
#ifndef NDEBUG
                print_current_automaton(my_merger.get(), "debug/model.",
                                        std::to_string(n_runs) + fmt::format(".f{:0>4}", uid++) + ".one_a");
                ref->doref(my_merger.get());
                performed_refinements.push_back(ref);
                print_current_automaton(my_merger.get(), "debug/model.",
                                        std::to_string(n_runs) + fmt::format(".f{:0>4}", uid++) + ".one_b");
                DLOG_S(2) << n_runs << ":one:" << printAddress(performed_refinements);
#else
                ref->doref(my_merger.get());
                performed_refinements.push_back(ref);
#endif
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
#ifndef NDEBUG
                print_current_automaton(my_merger.get(), "debug/model.",
                                        std::to_string(n_runs) + fmt::format(".f{:0>4}", uid++) + ".pro_a");
                ref->doref(my_merger.get());
                performed_refinements.push_back(ref);
                print_current_automaton(my_merger.get(), "debug/model.",
                                        std::to_string(n_runs) + fmt::format(".f{:0>4}", uid++) + ".pro_b");

                DLOG_S(2) << n_runs << ":uni:" << printAddress(performed_refinements);
#else
                ref->doref(my_merger.get());
                performed_refinements.push_back(ref);
#endif
            }
        }

        if (!no_isolated_states) {
            // There are still isolated states that need to be resolved.
            continue;
        }

#ifndef NDEBUG
        const std::size_t selected_refs_nr = performed_refinements.size();
        // printing the model each time
        print_current_automaton(my_merger.get(), "debug/model.", std::to_string(n_runs) + ".sel");
#endif

        // build hypothesis -> try to reduce the apta.
        active_learning_namespace::minimize_apta(performed_refinements, my_merger.get());

#ifndef NDEBUG
        DLOG_S(1) << n_runs << ": Got " << selected_refs_nr << " refs from selection and a total of "
                  << performed_refinements.size() << " after minimizing.";
        // printing the model each time
        print_current_automaton(my_merger.get(), "debug/model.", std::to_string(n_runs) + ".xxx");
#endif

        // Check equivalence, if not process the counter example.
        while (true) {
            /* While loop to check the type. type is < 0 if sul cannot properly respond to query, e.g. when the
            string we ask cannot be parsed in automaton. We ignore those cases, as they lead to extra states in
            hypothesis. This puts a burden on the equivalence oracle to make sure no query is asked twice, else we
            end up in infinite loop.*/

            optional<pair<vector<int>, int>> query_result = oracle->equivalence_query(my_merger.get(), teacher);
            if (!query_result) {
                LOG_S(INFO) << n_runs << ": Found consistent automaton => Print to " << OUTPUT_FILE;
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
            LOG_S(INFO) << n_runs << ": Counterexample of length " << cex.size() << " found: " << ss.str();
            proc_counterex(id, my_alphabet, cex, performed_refinements);
            break;
        }

        if (ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS) {
            LOG_S(INFO) << n_runs << ": Maximum of runs reached. Printing automaton to " << OUTPUT_FILE;
            active_learning_namespace::minimize_apta(performed_refinements, my_merger.get());
            print_current_automaton(my_merger.get(), OUTPUT_FILE, ".final");
            return;
        }
        ++n_runs;
    }
}
