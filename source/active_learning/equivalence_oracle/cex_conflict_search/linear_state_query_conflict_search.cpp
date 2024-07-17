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
 * @return int The response of the teacher.
 */
int dfa_conflict_search_namespace::linear_state_query_conflict_search::get_teacher_response(const vector<int>& cex, const std::unique_ptr<base_teacher>& teacher, inputdata& id) const {
    const pair< int, vector< vector<float> > > answer = teacher->get_membership_state_pair(cex, id);
    int resp = answer.first;
    return resp;
}