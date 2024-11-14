/**
 * @file weight_comparing_oracle.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-04-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "weight_comparing_oracle.h"
#include "common_functions.h"
#include "parameters.h"
#include "weight_comparator.h"

#include <cmath>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Returns true if the test trace is accepted, else 0.
 */
bool
weight_comparing_oracle::test_trace_accepted(apta& hypothesis, trace* const tr, inputdata& id){
    static const auto mu = static_cast<double>(MU);
    static const auto EOS = END_SYMBOL;

    apta_node* n = hypothesis.get_root();
    tail* t = tr->get_head();

    vector<int> current_substring;
    double sampled_value = 1;
    double true_value = 1;

    while (t != nullptr) {
        auto weights = get_weigth_distribution(current_substring, id);
        const int symbol = t->get_symbol();

        if (use_sinks && n==nullptr) {
            cout << "Sink found in eq" << endl;
            sampled_value = 0;

            const int mapped_symbol = stoi(id.get_symbol(symbol));
            true_value *= weights[mapped_symbol];
            if(true_value < mu) 
                return true;

            current_substring.push_back(symbol);
            t = t->future();

            continue;
        }
        else if (n == nullptr) {
            cout << "Counterexample because tree not parsable" << endl;
            return false;
        }

        if (t->is_final() && EOS != -1) {
            sampled_value *= static_cast<weight_comparator_data*>(n->get_data())->get_final_weight();
            true_value *= weights[EOS];
            break;
        } else if (t->is_final())
            break;

        sampled_value *= static_cast<weight_comparator_data*>(n->get_data())->get_weight(symbol);
        const int mapped_symbol = stoi(id.get_symbol(symbol));
        true_value *= weights[mapped_symbol];

        current_substring.push_back(symbol);
        n = active_learning_namespace::get_child_node(n, t);
        t = t->future();
    }

    auto diff = abs(true_value - sampled_value);
    if (diff > mu) {
        cout << "Predictions of the following counterexample: The true probability: " << true_value
            << ", predicted probability: " << sampled_value << endl;
        return false;
    }

    return true;
}

/**
 * @brief This function does what you think it does. The type does not matter, because we perform this one
 * on SULs that return us probabilities, hence it only returns a dummy value.
 *
 * Note: Requires that SOS and EOS be there.
 *
 * @param merger The merger.
 * @return std::optional< pair< vector<int>, int> > nullopt if no counterexample found, else the counterexample. The
 * type (integer) will always be zero.
 */
std::optional<pair<vector<int>, int>>
weight_comparing_oracle::equivalence_query(state_merger* merger) {
    inputdata& id = *(merger->get_dat());
    apta& hypothesis = *(merger->get_aut());

    static const auto& alphabet = id.get_alphabet();
    static const auto mu = static_cast<double>(MU);
    static const auto EOS = END_SYMBOL;

    // for debugging only
    static const bool CHECK_ACCESS_STRINGS = false;
    static const bool TEST_TARGET_VEC = false;

    if(CHECK_ACCESS_STRINGS){
        static unordered_set<apta_node*> tested_nodes;

        cout << "Testing the access strings" << endl;

        auto& alphabet = id.get_alphabet(); 
        for (red_state_iterator r_it = red_state_iterator(hypothesis.get_root()); *r_it != nullptr; ++r_it) {
            auto n = *r_it;

            while(n!=nullptr){
                
                if(tested_nodes.contains(n)){
                    n = n->get_next_merged();
                    continue;
                }
                
                // bypasses the root node, which is expected to be correct by default
                trace* at = n->get_access_trace();
                auto seq = at->get_input_sequence(true, true);

                for(auto symbol: alphabet){
                    seq[seq.size() - 1] = symbol;
                    trace* nt = vector_to_trace(seq, id);
                    if(!test_trace_accepted(hypothesis, nt, id)){
                        vector<int> query_string = at->get_input_sequence(true, false);
                        return make_optional<pair<vector<int>, int>>(make_pair(seq, 0));
                    }
                }
                tested_nodes.insert(n);
                n = n->get_next_merged();
            }
        }
    }

    if(TEST_TARGET_VEC){
        vector<int> test_vec;
        for(int i=2; i<=10; ++i){
            test_vec.push_back(id.get_reverse_symbol("314"));

            cout << "testing trace: ";
            for(int j=0; j<test_vec.size(); ++j){
                cout << id.get_symbol(test_vec[j]) << " ";
            }
            cout << endl;

            trace* nt = vector_to_trace(test_vec, id);
            if(!test_trace_accepted(hypothesis, nt, id)){
                return make_optional<pair<vector<int>, int>>(make_pair(test_vec, 0));
            }
        }
    }

    // here we start the actual search
    search_strategy->reset();
    std::optional<vector<int>> query_string_opt = search_strategy->next(id);
    while (query_string_opt != nullopt) {
        auto& query_string = query_string_opt.value();
        //for(int j=0; j<query_string.size(); ++j){
        //    cout << id.get_symbol(query_string[j]) << " ";
        //}
        //cout << endl;

        trace* test_tr = vector_to_trace(query_string, id, 0); // type-argument irrelevant here

        if(!test_trace_accepted(hypothesis, test_tr, id))
            return make_optional<pair<vector<int>, int>>(make_pair(query_string, 0));

        query_string_opt = search_strategy->next(id);
    }

    return nullopt;
}
