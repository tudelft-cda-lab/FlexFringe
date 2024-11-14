/**
 * @file active_state_sul_oracle.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-06-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "active_state_sul_oracle.h"
#include "common_functions.h"

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief We don't need the hidden states for the equivalence oracle, therefore 
 * we only fetch the output itself.
 * 
 * @return int The response of the SUL.
 */
int active_state_sul_oracle::get_sul_response(const vector<int>& query_string, inputdata& id) const {
    const pair< int, vector< vector<float> > > answer = get_membership_state_pair(query_string, id);
    int resp = answer.first;
    return resp;
}
