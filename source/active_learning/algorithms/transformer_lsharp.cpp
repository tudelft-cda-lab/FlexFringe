/**
 * @file transformer_lsharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-11-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "transformer_lsharp.h"
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

#include "weight_state_comparator.h"

#include <list>
#include <stdexcept>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Adds statistics to a node, returns the traces that were queried along the way.
 *
 * @param n The node to upate.
 * @return optional< vector<trace*> > A vector with all the new traces we added to the apta, or nullopt if we already
 * processed this node. Saves runtime.
 */
void transformer_lsharp_algorithm::query_weights(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                                              const vector<int>& alphabet, optional<pref_suf_t> seq_opt) const {

    static unordered_set<apta_node*> completed_nodes;
    if (completed_nodes.contains(n))
        return;
    static const int SOS = START_SYMBOL;
    static const int EOS = END_SYMBOL;

    auto* data = static_cast<weight_state_comparator_data*>(n->get_data());
    data->initialize_weights(alphabet);

    pref_suf_t seq;
    if (seq_opt) {
        seq = pref_suf_t(seq_opt.value());
    } else {
        auto access_trace = n->get_access_trace();
        seq = access_trace->get_input_sequence(true, false);
    }

    const pair< vector<float>, vector<float> > weight_state_pair = teacher->get_weigth_state_pair(seq, id);
    auto& weights = weight_state_pair.first;
    auto& state = weight_state_pair.second;

    for (int i = 0; i < weights.size(); ++i) {
        if (SOS > -1 && i == SOS)
            continue;
        else if (i == EOS) {
            data->set_final_weight(weights[i]);
            continue;
        }
        int symbol = id.symbol_from_string(to_string(i));
        data->set_weight(symbol, weights[i]);
    }

    if (n == merger->get_aut()->get_root()) {
        data->initialize_access_weight(weights[EOS]);
    } else {
        auto node_it = merger->get_aut()->get_root();
        double w_product = 1;
        for (auto s : seq) {
            data = static_cast<weight_state_comparator_data*>(node_it->get_data());
            w_product *= data->get_weight(s);
            node_it = node_it->get_child(s);
        }
        assert(node_it == n);
        data = static_cast<weight_state_comparator_data*>(n->get_data());
        w_product *= weights[EOS];
        data->initialize_access_weight(w_product);
    }

    data->initialize_state(state);

    completed_nodes.insert(n);
}

/**
 * @brief Main routine of this algorithm.
 *
 * @param id Inputdata.
 */
void transformer_lsharp_algorithm::run(inputdata& id) {
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
        query_weights(merger, root_node, id, alphabet, seq);
        extend_fringe(merger, root_node, the_apta, id, alphabet);
    }

    {
        static int model_nr = 0;
        print_current_automaton(merger.get(), "model.", "root");
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
