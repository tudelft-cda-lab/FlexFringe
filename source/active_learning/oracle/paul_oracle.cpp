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
paul_oracle::resp_t paul_oracle::get_sul_response(const vector< vector<int> >& query_string, inputdata& id) const {
    sul_response res = ask_sul(query_string, id);
    
    int type = res.GET_INT_VEC()[0];
    double confidence = res.GET_DOUBLE_VEC()[0];

    return resp_t(move(type), move(confidence));
}

/**
 * @brief Has a heuristic checking on whether we want to consider this string or not, but this 
 * time based on the transformer's confidence. If it is low omit this string, as it can 
 * introduce errors.
 */
bool paul_oracle::check_test_string_interesting(const double confidence) const noexcept {
    return confidence >= 0.9; // TODO: a bit heuristical. Can we do better?
}

/**
 * @brief This function does what you think it does.
 *
 * @param merger The merger.
 * @return std::optional< pair< vector<int>, int> > nullopt if no counterexample found, else the counterexample.
 */
std::optional<pair<vector<int>, sul_response>> paul_oracle::equivalence_query(state_merger* merger) {
    inputdata& id = *(merger->get_dat());
    apta& hypothesis = *(merger->get_aut());

    cex_search_strategy->reset();

    std::optional<vector<int>> query_string_opt = cex_search_strategy->next(id);
    while (query_string_opt != nullopt) { // nullopt == search exhausted
        if(!base_oracle::check_test_string_interesting(query_string_opt.value()))
            continue;

        if(tested_strings->contains(query_string_opt.value()))
            continue;
        tested_strings->add_suffix(query_string_opt.value());

        vector< vector<int> > query_string{ move(query_string_opt.value()) };
        const auto sul_resp = get_sul_response(query_string, id);
        if(!check_test_string_interesting(sul_resp.get_confidence()))
            continue;
        const int inferred_value = sul_resp.get_type();
    
        trace* test_tr = vector_to_trace(query_string[0], id, 0); // type-argument irrelevant here
        apta_node* n = hypothesis.get_root();
        tail* t = test_tr->get_head();

        for (int j = 0; j < test_tr->get_length(); j++) {
            n = active_learning_namespace::get_child_node(n, t);

            if (n == nullptr) {
                cout << "Counterexample because tree not parsable. Returning string and transformer prediction" << endl;
                return make_optional(make_pair(move(query_string[0]), sul_response(inferred_value)));
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
                return make_optional(make_pair(query_string[0], sul_response(inferred_value)));

            if(const auto& min_string = conflict_rep_pair.first; min_string.size() > AL_MAX_SEARCH_DEPTH){
                cout << "Found a counterexample whose size exceeds maximum search depth. Omitting this example" << endl;
                tested_strings->add_suffix(min_string);
                continue;
            }
            
            return make_optional(make_pair( move(conflict_rep_pair.first), conflict_rep_pair.second.value())); // TODO: this line bothers me for sure
        }

        query_string_opt = cex_search_strategy->next(id);
    }

    return nullopt;
}