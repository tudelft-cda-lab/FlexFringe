/**
 * @file ldot.h
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief The Ldot-algorithm, as part of my thesis
 * @version 0.1
 * @date 2024-1-9
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "ldot.h"
#include "loguru.hpp"
#include "sqldb_sul.h"
#include <fmt/format.h>

void ldot_algorithm::run(inputdata& id) {
    shared_ptr<sqldb_sul> my_sul = dynamic_pointer_cast<sqldb_sul>(sul);
    if (my_sul == nullptr) {
        throw logic_error("ldot only works with sqldb_sul.");
    }
    LOG_S(INFO) << "Running the ldot algorithm.";
    vector<int> trace = {1};
    int res = my_sul->query_trace_maybe(trace, id);
    std::string log = fmt::format("Getting {0}: {1}", my_sul->get_sqldb().vec2str(trace), res);
    LOG_S(INFO) << log;
    // TODO: Implement the thing!
}
