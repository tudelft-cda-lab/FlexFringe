/**
 * @file string_probability_oracle.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-04-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "string_probability_oracle.h"
#include "common_functions.h"
#include "parameters.h"
#include "string_probability_estimator.h"

#include <cmath>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief This function does what you think it does. The type does not matter, because we perform this one
 * on SULs that return us probabilities, hence it only returns a dummy value.
 *
 * TODO: this function is also inefficient in terms of inferring probabilities. Optimize this one.
 *
 * @param merger The merger.
 * @return std::optional< pair< vector<int>, int> > nullopt if no counterexample found, else the counterexample. The
 * type (integer) will always be zero.
 */
/* std::optional<pair<vector<int>, int>>
string_probability_oracle::equivalence_query(state_merger* merger) {
    inputdata& id = *(merger->get_dat());
    apta& hypothesis = *(merger->get_aut());

    static const auto& alphabet = id.get_alphabet();
    static const auto mu = static_cast<double>(MU);

    cex_search_strategy->reset();

    std::optional<vector<int>> query_string_opt = cex_search_strategy->next(id);
    while (query_string_opt != nullopt) { // nullopt == search exhausted
        auto& query_string = query_string_opt.value();

        double true_val = get_string_probability(query_string, id);

        if (true_val < 0)
            return make_optional<pair<vector<int>, int>>(
                query_string, true_val); // target automaton cannot be parsed with this query string

        trace* test_tr = vector_to_trace(query_string, id, 0); // type-argument irrelevant here

        apta_node* n = hypothesis.get_root();
        tail* t = test_tr->get_head();

        vector<int> current_substring;

        double sampled_probability = 1;
        while (t != nullptr) {
            if (n == nullptr) {
                cout << "Counterexample because tree not parsable" << endl;
                //cex_search_strategy->reset();
                return make_optional<pair<vector<int>, int>>(make_pair(query_string, 0));
            }

            if (t->is_final() && FINAL_PROBABILITIES) {
                sampled_probability *= static_cast<string_probability_estimator_data*>(n->get_data())->get_final_probability();
                break;
            } else if (t->is_final())
                break;

            const int symbol = t->get_symbol();
            sampled_probability *=
                static_cast<string_probability_estimator_data*>(n->get_data())->get_normalized_probability(symbol);

            n = active_learning_namespace::get_child_node(n, t);
            t = t->future();
        }

        auto diff = abs(true_val - sampled_probability);
        if (diff > mu) {
            cout << "Predictions of the following counterexample: The true probability: " << true_val
                 << ", predicted probability: " << sampled_probability << endl;
            //cex_search_strategy->reset();
            return make_optional<pair<vector<int>, int>>(make_pair(query_string, 0));
        }

        query_string_opt = cex_search_strategy->next(id);
    }

    return nullopt;
} */
