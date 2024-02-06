/**
 * @file transformer_lsharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
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

#include "types_state_comparator.h"

#include <list>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Take a node and complete it wrt to the alphabet.
 *
 */
void transformer_lsharp_algorithm::complete_state(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                                      const vector<int>& alphabet) const {
    static unordered_set<apta_node*> completed_nodes;
    if (completed_nodes.contains(n))
        return;

    for (const int symbol : alphabet) {
        if (n->get_child(symbol) == nullptr) {
            auto access_trace = n->get_access_trace();

            pref_suf_t seq;
            if (n->get_number() != -1)
                seq = access_trace->get_input_sequence(true, true);
            else
                seq.resize(1);

            seq[seq.size() - 1] = symbol;

            const pair<int, vector<float>> answer = teacher->get_membership_state_pair(seq, id);
            const int type = answer.first;
            const auto& hidden_state = answer.second;

            trace* new_trace = vector_to_trace(seq, id, type);
            id.add_trace_to_apta(new_trace, merger->get_aut(), false);
            id.add_trace(new_trace);

            // add the hidden state as well
            auto child_node_data = dynamic_cast<type_state_comparator_data*>(n->get_child(symbol)->get_data());
            child_node_data->initialize_state(hidden_state);
        }
    }
    completed_nodes.insert(n);
}