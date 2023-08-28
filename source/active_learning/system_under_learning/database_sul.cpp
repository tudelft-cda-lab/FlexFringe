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

void database_sul::pre(inputdata& id) {
}

bool database_sul::is_member(const std::vector<int>& query_trace) const {
  return database->is_member(query_trace);
}

/**
 * @brief We don't need this function in here at the moment.
 * 
 * @param query_trace 
 * @param id 
 * @return const int 
 */
const int database_sul::query_trace(const std::vector<int>& query_trace, inputdata& id) const {
  return true; // return database->query_trace(query_trace, id);
}