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

        if(!base_oracle::check_test_string_interesting(query_string_opt.value()) || tested_strings->contains(query_string_opt.value())){
            query_string_opt = cex_search_strategy->next(id);
            continue;
        }
        tested_strings->add_suffix(query_string_opt.value());
        
        vector< vector<int> > query_string{ move(query_string_opt.value()) };
        const auto sul_resp = get_sul_response(query_string, id);
        if(!check_test_string_interesting(sul_resp.get_confidence())){
            cout << "Skipped string due to low confidence of transformer" << endl;
            query_string_opt = cex_search_strategy->next(id);
            continue; 
        }

        pair<bool, optional<sul_response> > resp = conflict_detector->creates_conflict(query_string, hypothesis, id);

        if (resp.first) {
            pair< vector<int>, sul_response> conflict_resp_pair = conflict_searcher->get_conflict_string(query_string, hypothesis, id);
            
            cout << "Found counterexample of length " << conflict_resp_pair.first.size() << ":";
            for(auto x: conflict_resp_pair.first)
                cout << " " << id.get_symbol(x);
            cout << endl;
            
            return make_optional(conflict_resp_pair);
        }

        query_string_opt = cex_search_strategy->next(id);
    }

    return nullopt;
}