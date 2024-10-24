/**
 * @file teacher_imat.cpp
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-04-12
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "teacher_imat.h"
#include "definitions.h"

using namespace active_learning_namespace;

/**
 * @brief Use query_trace_maybe to also get unknown (-1)
 *
 * @param query
 * @param id
 * @return const int 0 or greater for the type.
 */
const int teacher_imat::ask_membership_query(const pref_suf_t& query, inputdata& id) {
    return sul->query_trace_maybe(query, id);
}
