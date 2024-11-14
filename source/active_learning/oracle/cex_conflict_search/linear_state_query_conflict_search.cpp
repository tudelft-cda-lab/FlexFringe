/**
 * @file linear_state_query_conflict_search.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "linear_state_query_conflict_search.h"

using namespace std;

/**
 * @brief We don't need the hidden states for the equivalence oracle, therefore 
 * we only fetch the output itself.
 * 
 * @return int The response of the oracle.
 */
const int dfa_conflict_search_namespace::linear_state_query_conflict_search::get_teacher_response(const vector<int>& substr, const std::unique_ptr<oracle_base>& oracle, inputdata& id) const {
    const sul_response response = oracle->ask_sul(substr, id);
    int resp = response.GET_INT();
    return resp;
}