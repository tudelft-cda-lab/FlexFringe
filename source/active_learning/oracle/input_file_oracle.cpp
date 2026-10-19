/**
 * @file input_file_oracle.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-03-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "input_file_oracle.h"
#include "common_functions.h"
#include "predict_mode.h"

#include <optional>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Checks if the hypothesis is consistent with the set of input traces that are stores in 
 * input_file_sul.
 */
optional<pair<vector<int>, sul_response>> input_file_oracle::equivalence_query(state_merger* merger) {
    const count_driven* const eval_func = dynamic_cast<count_driven*>(merger->get_eval());
    if (eval_func == nullptr)
        throw logic_error("Must have a heuristic that derives from count_driven at this point.");

    for (const auto& [sequence, type] : dynamic_cast<input_file_sul*>(sul.get())->get_all_traces()) {
        trace* tr = vector_to_trace(sequence, *(merger->get_dat()), type);
        if (!active_learning_namespace::aut_accepts_trace(tr, merger->get_aut(), eval_func)) {
            return make_optional(make_pair(sequence, sul_response(type)));
        }
    }
    return nullopt;
}