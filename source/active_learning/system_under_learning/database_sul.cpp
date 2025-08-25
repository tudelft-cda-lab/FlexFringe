/**
 * @file database_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-06-23
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "database_sul.h"

using namespace std;

void database_sul::pre(inputdata& id) {}

const sul_response database_sul::do_query(const vector<int>& query_trace, inputdata& id) const { 
    return database->is_member(query_trace) ? sul_response(true) : sul_response(false); 
}