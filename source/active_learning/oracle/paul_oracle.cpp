/**
 * @file paul_oracle.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "paul_oracle.h"
#include "common_functions.h"

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Avoids duplicate code in the active_state_sul_oracle.
 * 
 * @return int The response of the SUL.
 */
/* int paul_oracle::get_sul_response(const vector< vector<int> >& query_string, inputdata& id) const {
    sul_response res = ask_sul(query_string, id);
    
    vector< pair<int, float> >

    int type = res.GET_INT_VEC()[0];
    float confidence = res.GET_FLOAT_VEC()[0];

    return type;
} */

/**
 * @brief This function does what you think it does.
 *
 * @param merger The merger.
 * @return std::optional< pair< vector<int>, int> > nullopt if no counterexample found, else the counterexample.
 */
/* std::optional<pair<vector<int>, sul_response>> paul_oracle::equivalence_query(state_merger* merger) {
    // TODO: this one is going to look a little more complicated perhaps. 
    throw logic_error("EQ not implemented for PAUL oracle yet Perhaps do a sort of overlap driven conflict detector?");

    inputdata& id = *(merger->get_dat());
    apta& hypothesis = *(merger->get_aut());

    cex_search_strategy->reset();

    std::optional<vector<int>> query_string_opt = cex_search_strategy->next(id);
    while (query_string_opt != nullopt) { // nullopt == search exhausted
        vector< vector<int> > query_string{ move(query_string_opt.value()) };
        int inferred_value = get_response(query_string, id);
        trace* test_tr = vector_to_trace(query_string[0], id, 0); // type-argument irrelevant here

        apta_node* n = hypothesis.get_root();
        tail* t = test_tr->get_head();
        for (int j = 0; j < test_tr->get_length(); j++) {
            n = active_learning_namespace::get_child_node(n, t);

            if (n == nullptr) {
                cout << "Counterexample because tree not parsable. Returning string and transformer prediction" << endl;
                return make_optional<pair<vector<int>, int>>(make_pair(move(query_string[0]), inferred_value));
            }

            t = t->future();
        }

        const int pred_val = n->get_data()->predict_type(t);
        if (inferred_value != pred_val) {
            cout << "Found counterexample. The inferred value: " << id.get_type(inferred_value)
                 << ", predicted: " << id.get_type(pred_val) << endl;
            cout << "String before conflict resolution: ";
            for(int x: query_string[0])
                cout << id.get_symbol(x) << " ";
            cout << endl;
            
            pair< vector<int>, optional<sul_response> > conflict_rep_pair = conflict_searcher->get_conflict_string(query_string[0], hypothesis, id);
            if(conflict_rep_pair.second == nullopt)
                return make_optional(make_pair(query_string[0], inferred_value));
            return make_optional(make_pair( move(conflict_rep_pair.first), conflict_rep_pair.second.value())); // TODO: this line bothers me for sure
        }

        query_string_opt = cex_search_strategy->next(id);
    }

    return nullopt;
}
 */