/**
 * @file transformer_weighted_lsharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-11-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "transformer_weighted_lsharp.h"
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
void transformer_weighted_lsharp_algorithm::query_weights(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
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