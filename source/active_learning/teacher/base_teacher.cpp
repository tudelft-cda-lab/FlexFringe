/**
 * @file base_teacher.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-02-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "base_teacher.h"

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Asks the teacher a membership query. In case the automaton cannot be parsed with the query, it returns -1.
 * Else the mapping as defined by the inputdata.
 *
 * @param prefix S.e.
 * @param suffix S.e.
 * @param id Inputdata.
 * @return const int 0 or greater for the type, -1 if automaton cannot be parsed with the query.
 */
const int base_teacher::ask_membership_query_lstar(const pref_suf_t& prefix, const pref_suf_t& suffix, inputdata& id) {
    pref_suf_t query(prefix);
    query.insert(query.end(), suffix.begin(), suffix.end());

    return ask_membership_query(query, id);
}

/**
 * @brief See ask_membership_query_lstar(const pref_suf_t& prefix, const pref_suf_t& suffix, inputdata& id)
 *
 * @param query
 * @param id
 * @return const int 0 or greater for the type, -1 if automaton cannot be parsed with the query.
 */
const int base_teacher::ask_membership_query(const pref_suf_t& query, inputdata& id) {
    return sul->query_trace(query, id);
}

const std::pair<int, std::vector<float>>
base_teacher::get_membership_state_pair(const active_learning_namespace::pref_suf_t& access_seq, inputdata& id) {
    return sul->get_type_and_state(access_seq, id);
}

/**
 * @brief Gets the probability of a certain string. Only supported with certain SULs.
 *
 * @param query
 * @param id
 * @return const double
 */
const double base_teacher::get_string_probability(const pref_suf_t& query, inputdata& id) {
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
const std::vector<float> base_teacher::get_weigth_distribution(const active_learning_namespace::pref_suf_t& access_seq,
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
const std::pair<std::vector<float>, std::vector<float>>
base_teacher::get_weigth_state_pair(const active_learning_namespace::pref_suf_t& access_seq, inputdata& id) {
    return sul->get_weights_and_state(access_seq, id);
}
