/**
 * @file probabilistic_lsharp_v2.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-10-09
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "probabilistic_lsharp_v2.h"
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

#include "string_probability_estimator_v2.h"

#include <list>
#include <stdexcept>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Take a node and complete it wrt to the alphabet.
 *
 * This is the only function that creates new nodes (apart from the root node).
 *
 */
unordered_set<apta_node*> probabilistic_lsharp_v2_algorithm::extend_fringe(unique_ptr<state_merger>& merger, apta_node* n,
                                                                        unique_ptr<apta>& the_apta, inputdata& id,
                                                                        const vector<int>& alphabet) const {
    static unordered_set<apta_node*> extended_nodes; // TODO: we might not need this, since it is inherently backed in
                                                     // add_statistics and the size of traces
    if (extended_nodes.contains(n))
        return unordered_set<apta_node*>();

    unordered_set<apta_node*> new_nodes;

    // this path is only relevant in traces that were added via counterexample processing
    auto access_trace = n->get_access_trace();

    pref_suf_t seq;
    if (n->get_number() != -1 && n->get_number() != 0)
        seq = access_trace->get_input_sequence(true, true);
    else
        seq.resize(1);

    for (const int symbol : alphabet) {
        seq[seq.size() - 1] = symbol;
        trace* new_trace = vector_to_trace(seq, id);
        id.add_trace_to_apta(new_trace, merger->get_aut(), false);

        add_statistics(merger, n->get_child(symbol), id, alphabet, seq);
        new_nodes.insert(n->get_child(symbol));
    }
    return new_nodes;
}

/**
 * @brief Does what you think it does.
 */
void probabilistic_lsharp_v2_algorithm::init_final_prob(apta_node* n, apta* the_apta, inputdata& id) const {
    pref_suf_t seq;

    [[likely]] if (n->get_number() != -1 && n->get_number() != 0) {
        auto at = n->get_access_trace();
        seq = at->get_input_sequence(true, false);
    }

    trace* new_trace = vector_to_trace(seq, id);
    const double new_prob = teacher->get_string_probability(seq, id);
    static_cast<string_probability_estimator_v2_data*>(n->get_data())->init_access_probability(new_prob);
}

/**
 * @brief Adds statistics to a node, returns the traces that were queried along the way.
 *
 * @param n The node to upate.
 * @return optional< vector<trace*> > A vector with all the new traces we added to the apta, or nullopt if we already
 * processed this node. Saves runtime.
 */

/**
 * @brief Adds statistics to a node, returns the traces that were queried along the way.
 * 
 * @param merger 
 * @param n 
 * @param id 
 * @param alphabet 
 * @param seq_opt The sequence leading to the node, done through optimization. Empty optional only for the root node.
 * @param p_mass The remaining probability mass to update this branch.
 */
void probabilistic_lsharp_v2_algorithm::add_statistics(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                                                    const vector<int>& alphabet, optional<pref_suf_t> seq_opt) const {
    static unordered_set<apta_node*> completed_nodes;
    if (completed_nodes.contains(n))
        return;

    auto* data = static_cast<string_probability_estimator_v2_data*>(n->get_data());
    data->initialize_distributions(alphabet);

    pref_suf_t seq;
    if (seq_opt) {
        seq = pref_suf_t(seq_opt.value());
        seq.resize(seq_opt.value().size() + 1);
    } else {
        auto access_trace = n->get_access_trace();
        seq = access_trace->get_input_sequence(true, true);
    }

    for (const int symbol : alphabet) {
        seq[seq.size() - 1] = symbol;
        trace* new_trace = vector_to_trace(seq, id);
        id.add_trace(new_trace);

        const double new_prob = teacher->get_string_probability(
            seq, id); // get_probability_of_last_symbol(new_trace, id, teacher, merger->get_aut());
        if (std::isnan(new_prob))
            throw runtime_error("Error: NaN value has occurred."); // debugging
        /* cout << "Seq:";
        print_vector(seq);
        cout << "prob: " << new_prob << endl; */

        data->update_probability(symbol, new_prob);
    }



    init_final_prob(n, merger->get_aut(), id);
    update_access_path(n, merger->get_aut(), alphabet); // must be called after init_final_prob()

    completed_nodes.insert(n);
}

/**
 * @brief Takes a node n, and updates all the probabilities that lead to it.
 *
 * This function must only be called from within add_statistics() and
 * must be called after init_final_prob() [make sure that the node n's
 * final probability has been initialized].
 *
 * @param n The node.
 * @param the_apta The apta.
 */
void probabilistic_lsharp_v2_algorithm::update_access_path(apta_node* n, apta* the_apta,
                                                        const vector<int>& alphabet) const {
    auto access_trace = n->get_access_trace();
    if (access_trace == nullptr)
        return; // The root node

    stack<apta_node*> nodes_to_update;
    apta_node* current_node = the_apta->get_root();

    tail* t = access_trace->get_head();

    // TODO: we can simplify that one
    while (current_node != n) { // walk all the way to n, getting the states up to there
        nodes_to_update.push(current_node);
        current_node = current_node->get_child(t->get_symbol());
        t = t->future();
    }

    // update the nodes behind n
    const auto& data_to_add =
        static_cast<string_probability_estimator_v2_data*>(n->get_data())->get_outgoing_distribution();
    while (!nodes_to_update.empty()) {
        current_node = nodes_to_update.top();
        if (current_node == n)
            continue;

        auto current_node_data = static_cast<string_probability_estimator_v2_data*>(current_node->get_data());

        for (int s = 0; s < data_to_add.size(); ++s) { current_node_data->add_probability(s, data_to_add[s]); }

        nodes_to_update.pop();
    }
}

/**
 * @brief DFS procedure through the tree. Updates the probabilities according to the new fringe and normalizes.
 * Return value indicates whether a final probability of a node has been larger than one. In case it has been
 * we return true, else false.
 *
 * @return true A node has final prob larger than 1.
 * @return false No final prob larger than 1.
 */
/* bool */ void probabilistic_lsharp_v2_algorithm::update_tree_dfs(apta* the_apta, const vector<int>& alphabet) const {
    stack<double> p_stack;
    stack<double> remaining_mass_stack;

    apta_node* n = the_apta->get_root();
    auto data = static_cast<string_probability_estimator_v2_data*>(n->get_data());

    auto estimated_masses = string_probability_estimator_v2::normalize_probabilities(
        static_cast<string_probability_estimator_v2_data*>(n->get_data()), 1, alphabet.size());

    unordered_set<apta_node*> node_set;
    stack<apta_node*> node_stack;

    node_set.insert(n);
    for (auto s : alphabet) {
        auto child_node = n->get_child(s);
        if (node_set.contains(child_node) || child_node == nullptr)
            continue;

        node_stack.push(child_node);
        p_stack.push(static_cast<string_probability_estimator_v2_data*>(n->get_data())->get_normalized_probability(s));
        remaining_mass_stack.push(estimated_masses[s]);
    }

    while (!node_stack.empty()) {
        n = node_stack.top();
        auto p = p_stack.top();
        auto remaining_mass = remaining_mass_stack.top();
        node_stack.pop();
        p_stack.pop();
        remaining_mass_stack.pop();

        auto data = static_cast<string_probability_estimator_v2_data*>(n->get_data());
        /* bool large_fp =  */ data->update_final_prob(p);
        /* if(large_fp){
          cout << "Found final probability larger than 1. Extending tree." << endl;
          return true;
        } */
        auto estimated_masses = string_probability_estimator_v2::normalize_probabilities(data, remaining_mass, alphabet.size());

        for (auto s : alphabet) {
            auto child_node = n->get_child(s);
            if (node_set.contains(child_node) || child_node == nullptr)
                continue;

            node_stack.push(child_node);
            double new_p =
                p * static_cast<string_probability_estimator_v2_data*>(n->get_data())->get_normalized_probability(s);
            p_stack.push(new_p);
            remaining_mass_stack.push(estimated_masses[s]);
        }

        node_set.insert(n);
    }

    /* return false; */
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
void probabilistic_lsharp_v2_algorithm::proc_counterex(const unique_ptr<base_teacher>& teacher, inputdata& id,
                                                    unique_ptr<apta>& hypothesis, const vector<int>& counterex,
                                                    unique_ptr<state_merger>& merger, const refinement_list refs,
                                                    const vector<int>& alphabet) const {
    // linear search to find fringe, then append new states
    cout << "proc counterex" << endl;
    reset_apta(merger.get(), refs);

    vector<int> substring;
    apta_node* n = hypothesis->get_root();
    for (auto s : counterex) {
        if (n == nullptr) {
            const auto queried_type = teacher->ask_membership_query(substring, id); // TODO: necessary here?
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
    const auto queried_type = teacher->ask_membership_query(substring, id); // TODO: necessary here?
    trace* new_trace = vector_to_trace(substring, id, queried_type);
    id.add_trace_to_apta(new_trace, hypothesis.get(), false);
    id.add_trace(new_trace);

    // now let's walk over the apta again, completing all the states we created
    n = hypothesis->get_root();
    trace* parsing_trace = vector_to_trace(counterex, id);
    tail* t = parsing_trace->get_head();
    // double product = 1;

    while (n != nullptr) {
        add_statistics(merger, n, id, alphabet, nullopt);
        // if(queried_traces) extend_fringe(merger, n, hypothesis, id, queried_traces.value());

        // auto* data = static_cast<string_probability_estimator_v2_data*>(n->get_data());
        // product *= data->get_weight(t->get_symbol());

        n = active_learning_namespace::get_child_node(n, t);
        t = t->future();

        // if(n != nullptr) static_cast<string_probability_estimator_v2_data*>(n->get_data())->update_final_prob(product);
    }
}

/**
 * @brief Does what you think it does.
 *
 * @param merger
 * @param the_apta
 * @return list<refinement*>
 */
list<refinement*> probabilistic_lsharp_v2_algorithm::find_complete_base(unique_ptr<state_merger>& merger,
                                                                     unique_ptr<apta>& the_apta, inputdata& id,
                                                                     const vector<int>& alphabet) {
    static int depth = 1;           // because we initialize root node with depth 1
    static const int MAX_DEPTH = 6; // 6 for most problems

    list<refinement*> performed_refs;
    unordered_set<apta_node*> fringe_nodes;

    int merge_depth = 0;

    // if(MAX_DEPTH_REACHED){
    //   minimize_apta(performed_refs, merger.get());
    //   return performed_refs;
    // }

    while (true) { // cancel when either not red node identified or max depth

        unordered_set<apta_node*> blue_nodes;
        for (blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it) {
            auto blue_node = *b_it;
            blue_nodes.insert(blue_node);
        }

        bool reached_fringe = blue_nodes.empty();
        if (reached_fringe /*  || underestimated_dist */) {
            cout << "Reached fringe. Extend and recompute merges" << endl;
            reset_apta(merger.get(), performed_refs);
            performed_refs.clear();

            /* std::unordered_set<apta_node*> created_nodes; */
            for (auto blue_node : fringe_nodes) {
                auto cn = extend_fringe(merger, blue_node, the_apta, id, alphabet);
                /* for(auto new_node: cn) created_nodes.insert(new_node); */
            }
            fringe_nodes.clear();
            /* underestimated_dist =  */ update_tree_dfs(the_apta.get(), alphabet);
            /* if(underestimated_dist) */
            /* fringe_nodes = created_nodes; */

            merge_depth = 0;
            ++depth;
            continue;
        } else if (depth == MAX_DEPTH) {
            cout << "Max-depth reached. Printing the automaton. Depth:" << depth << endl;
            // minimize_apta(performed_refs, merger.get());
            find_closed_automaton(performed_refs, the_apta, merger, string_probability_estimator_v2::get_distance);
            cout << "Minimized. Now printing" << endl;
            // MAX_DEPTH_REACHED = true;
            // return performed_refs;
            print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
            cout << "Printed. Terminating" << endl;
            exit(0);
        }

        // go through each newly found fringe node, see if you can merge or extend
        ++merge_depth;
        if (merge_depth <= depth)
            fringe_nodes.clear();

        bool identified_red_node = false;
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

        {
            static int model_nr = 0;
            print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_refs");
        }

        if (!identified_red_node) {
            cout << "Complete basis found. Forwarding hypothesis" << endl;
            return performed_refs;
        } else if (!reached_fringe) {
            cout << "Processed layer " << merge_depth << endl;
            // fringe_nodes.clear();
            continue;
        }
    }
}

/**
 * @brief Main routine of this algorithm.
 *
 * @param id Inputdata.
 */
void probabilistic_lsharp_v2_algorithm::run(inputdata& id) {
    MERGE_WHEN_TESTING = true;
    cout << "testmerge option set to true, because algorithm relies on it" << endl;

    int n_runs = 1;

    auto eval = unique_ptr<evaluation_function>(get_evaluation());
    eval->initialize_before_adding_traces();

    auto the_apta = unique_ptr<apta>(new apta());
    auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));

    const vector<int> alphabet = id.get_alphabet();
    cout << "Alphabet: ";
    active_learning_namespace::print_sequence<vector<int>::const_iterator>(alphabet.cbegin(), alphabet.cend());

    {
        // init the root node
        auto root_node = the_apta->get_root();
        pref_suf_t seq;
        add_statistics(merger, root_node, id, alphabet, seq);
        extend_fringe(merger, root_node, the_apta, id, alphabet);
        update_tree_dfs(the_apta.get(), alphabet);
        // test_dfs(the_apta.get(), alphabet, teacher, id);
    }

    while (ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS) {
        if (n_runs % 100 == 0)
            cout << "Iteration " << n_runs + 1 << endl;

        auto refs = find_complete_base(merger, the_apta, id, alphabet);

        cout << "Searching for counterexamples" << endl;

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
            /* underestimated_dist =  */ update_tree_dfs(the_apta.get(),
                                                         alphabet); // TODO: this can be a more efficient update

            break;
        }

        // TODO: this one might not hold
        ++n_runs;
        if (ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS) {
            cout << "Maximum of runs reached. Printing automaton." << endl;
            print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
            return;
        }
    }
}
