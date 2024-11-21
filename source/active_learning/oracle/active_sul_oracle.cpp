/**
 * @file active_sul_oracle.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-04-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "active_sul_oracle.h"
#include "common_functions.h"

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief This function does what you think it does.
 *
 * @param merger The merger.
 * @return std::optional< pair< vector<int>, int> > nullopt if no counterexample found, else the counterexample.
 */
std::optional<pair<vector<int>, sul_response>> active_sul_oracle::equivalence_query(state_merger* merger) {
    inputdata& id = *(merger->get_dat());
    apta& hypothesis = *(merger->get_aut());

    cex_search_strategy->reset();

    std::optional<vector<int>> query_string_opt = cex_search_strategy->next(id);
    while (query_string_opt != nullopt) { // nullopt == search exhausted
        auto& query_string = query_string_opt.value();
        sul_response resp = sul->do_query(query_string, id);
        int true_val = resp.GET_INT();

        if (true_val < 0)
            return make_optional(query_string, resp); // target automaton cannot be parsed with this query string

        trace* test_tr = vector_to_trace(query_string, id, 0); // type-argument irrelevant here

        apta_node* n = hypothesis.get_root();
        tail* t = test_tr->get_head();
        for (int j = 0; j < test_tr->get_length(); j++) {
            n = active_learning_namespace::get_child_node(n, t);

            if (n == nullptr) {
                cout << "Counterexample because tree not parsable" << endl;
                //cex_search_strategy->reset();
                return make_optional<pair<vector<int>, int>>(make_pair(query_string, true_val));
            }

            t = t->future();
        }
        const int pred_val = n->get_data()->predict_type(t);
        if (true_val != pred_val) {
            cout << "Found counterexample. The true value: " << id.get_type(true_val)
                 << ", predicted: " << id.get_type(pred_val) << endl;
            cout << "String before conflict resolution: ";
            for(auto x: query_string)
                cout << id.get_symbol(x) << " ";
            cout << endl;
            
            pair< vector<int>, optional<response_wrapper> > conflict_rep_pair = conflict_searcher->get_conflict_string(query_string, hypothesis, id);
            if(conflict_rep_pair.second == nullopt)
                return make_optional<pair<vector<int>, int>>(make_pair(query_string, true_val));
            return make_optional<pair<vector<int>, int>>(make_pair(conflict_rep_pair.first, conflict_rep_pair.second.value().get_int_response()));
        }

        query_string_opt = cex_search_strategy->next(id);
    }

    return nullopt;
}
