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
#include <fmt/format.h>

void ldot_algorithm::run(inputdata& id) {
    LOG_S(INFO) << "Running the ldot algorithm.";
    vector<int> trace = {1};
    int res = sul->query_trace_maybe(trace, id);
    std::string log = fmt::format("Getting {0}: {1}", sul->get_sqldb().vec2str(trace), res);
    LOG_S(INFO) << log;
    // TODO: Implement the thing!
}
