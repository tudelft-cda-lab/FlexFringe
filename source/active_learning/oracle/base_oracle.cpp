/**
 * @file base_oracle.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "base_oracle.h"
#include "definitions.h"
#include "common_functions.h"

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Giving the strategy access to the merger. 
 * 
 * Was hard to put into the constructors at time of writing.
 */
void base_oracle::initialize(state_merger* merger){
    cex_search_strategy->initialize(merger);
}

const sul_response base_oracle::ask_sul(const vector<int>& query_trace, inputdata& id) const {
    return sul->do_query(query_trace, id);
}

const sul_response base_oracle::ask_sul(const vector<int>& prefix, const vector<int>& suffix, inputdata& id) const {
    vector<int> query_trace = concatenate_vectors(prefix, suffix);
    return ask_sul(query_trace, id);
}

const sul_response base_oracle::ask_sul(const vector<int>&& query_trace, inputdata& id) const {
    return sul->do_query(query_trace, id);
}

const sul_response base_oracle::ask_sul(const vector<int>&& prefix, const vector<int>&& suffix, inputdata& id) const {
    vector<int> query_trace = concatenate_vectors(prefix, suffix);
    return ask_sul(move(query_trace), id);
}

const sul_response base_oracle::ask_sul(const vector< vector<int> >& query_traces, inputdata& id) const {
    return sul->do_query(query_traces, id);
}

const sul_response base_oracle::ask_sul(const vector< vector<int> >&& query_traces, inputdata& id) const {
    return sul->do_query(query_traces, id);
}

/**
 * @brief Has a heuristic checking on whether we want to consider this string or not. Returns true if string interesting, 
 * false if we do not want to check.
 */
bool base_oracle::check_test_string_interesting(const vector<int>& teststr) const noexcept {
    if(!AL_TEST_EMTPY_STRING && teststr.size() == 0){
        return false;
    }
    return true;
}

/**
 * @brief The basic flow of the equivalence oracle. Can be overwritten in some exceptional 
 * examples, but most oracle implementations will use this flow.
 *
 * @param merger The state-merger.
 * @return std::optional< std::pair< std::vector<int>, int> > Counterexample if not equivalent, else nullopt.
 * Counterexample is pair of trace and the answer to the counterexample as returned by the SUL.
 */
optional< pair<vector<int>, sul_response> > base_oracle::equivalence_query(state_merger* merger) {
    inputdata& id = *(merger->get_dat());
    apta& hypothesis = *(merger->get_aut());

    cex_search_strategy->reset();

    optional<vector<int>> query_string_opt = cex_search_strategy->next(id);
    while (query_string_opt != nullopt) { // nullopt == search exhausted
        auto& query_string = query_string_opt.value();
        if(!check_test_string_interesting(query_string))
            continue;

        pair<bool, optional<sul_response> > resp = conflict_detector->creates_conflict(query_string, hypothesis, id);
        if(resp.first){
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