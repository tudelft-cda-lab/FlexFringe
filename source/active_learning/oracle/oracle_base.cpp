/**
 * @file oracle_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "oracle_base.h"
#include "definitions.h"

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Asks the oracle a membership query. In case the automaton cannot be parsed with the query, it returns -1.
 * Else the mapping as defined by the inputdata.
 *
 * @param prefix S.e.
 * @param suffix S.e.
 * @param id Inputdata.
 * @return const int 0 or greater for the type, -1 if automaton cannot be parsed with the query.
 */
const int base_oracle::ask_membership_query(const pref_suf_t& prefix, const pref_suf_t& suffix, inputdata& id) {
    pref_suf_t query(prefix);
    query.insert(query.end(), suffix.begin(), suffix.end());

    return ask_membership_query(query, id);
}

/**
 * @brief See ask_membership_query(const pref_suf_t& prefix, const pref_suf_t& suffix, inputdata& id)
 *
 * @return const int 0 or greater for the type, -1 if target cannot answer the query.
 */
const int base_oracle::ask_membership_query(const pref_suf_t& query, inputdata& id) {
    return sul->query_trace(query, id);
}

/**
 * @brief See ask_membership_query(const pref_suf_t& prefix, const pref_suf_t& suffix, inputdata& id)
 *
 * @return Integer in pair like overloaded function. Float represents confidence of oracle (e.g. transformer) that output is correct. 
 */
const pair<int, float> base_oracle::ask_membership_confidence_query(const pref_suf_t& query, inputdata& id) {
    tuple<int, float, vector< vector<float> >> query_res = sul->get_type_confidence_and_states(query, id);
    return make_pair(get<0>(query_res), get<1>(query_res));
}

const std::vector< std::pair<int, float> > 
base_oracle::ask_type_confidence_batch(const std::vector< std::vector<int> >& query_traces, inputdata& id) const {
    return sul->get_type_confidence_batch(query_traces, id);
}


/**
 * @brief 
 * 
 * @return const std::pair< int, std::vector< std::vector<float> > > Int is response of network, vector<float> is the sequence of hidden states.
 */
const std::pair< int, std::vector< std::vector<float> > > 
base_oracle::get_membership_state_pair(const active_learning_namespace::pref_suf_t& access_seq, inputdata& id){
    return sul->get_type_and_states(access_seq, id);
}

/**
 * @brief Gets the probability of a certain string. Only supported with certain SULs.
 *
 * @param query
 * @param id
 * @return const double
 */
const double base_oracle::get_string_probability(const pref_suf_t& query, inputdata& id) {
    double res = sul->get_string_probability(query, id);
    return res;
}

/**
 * @brief Gets the weights from the state reached by access_seq.
 *
 * @param access_seq
 * @param id
 * @return const std::vector<float>
 */
const std::vector<float> base_oracle::get_weigth_distribution(const active_learning_namespace::pref_suf_t& access_seq,
                                                               inputdata& id) {
    return sul->get_weight_distribution(access_seq, id);
}

/**
 * @brief Get the weigth state pair object
 * 
 * @param access_seq 
 * @param id 
 * @return const std::pair< std::vector<float>, std::vector<float> > 
 */
const std::pair< std::vector<float>, std::vector<float> > base_oracle::get_weigth_state_pair(const active_learning_namespace::pref_suf_t& access_seq,
                                                     inputdata& id) {
    return sul->get_weights_and_state(access_seq, id);
}

/**
 * @brief For ldot
 * 
 * @param query 
 * @param id 
 * @return const int 
 */
const int oracle_base::ask_membership_query_maybe(const pref_suf_t& query, inputdata& id) {
    return sul->query_trace_maybe(query, id);
}

const sul_response ask_sul(const std::vector<int>& query_trace, inputdata& id) const {
    return sul->query_trace(query_trace, id);
}


const sul_response ask_sul(const std::vector<int>&& query_trace, inputdata& id) const {
    return sul->query_trace(query_trace, id);
}

const sul_response ask_sul(const std::vector< std::vector<int> >& query_traces, inputdata& id) const {
    return sul->query_trace(query_traces, id);
}

const sul_response ask_sul(const std::vector< std::vector<int> >&& query_traces, inputdata& id) const {
    return sul->query_trace(query_traces, id);
}