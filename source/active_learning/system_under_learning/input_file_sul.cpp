/**
 * @file input_file_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-02-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "input_file_sul.h"
#include "abbadingoparser.h"
#include "csvparser.h"
#include "inputdatalocator.h"
#include "parameters.h"

#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Returns the type according to the input data, and if not possible returns an extra unknown type.
 */
const sul_response input_file_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
    return sul_response(all_traces.at(query_trace));
}

/**
 * @brief Initializes the set of all traces, and initializes the unknown type string.
 */
void input_file_sul::pre(inputdata& id) {
    for (const auto it : id) {
        trace& current_trace = *it;
        const auto current_sequence = current_trace.get_input_sequence();
        if (all_traces.contains(current_sequence))
            continue;

        const int type = current_trace.get_type();
        all_traces[current_sequence] = type;
    }

    // add the unknown type to the set of types. 
    string unk_t_string = "_unk";
    int counter = 0;
    const auto& r_types = id.get_r_types();
    while (r_types.contains(unk_t_string)) { unk_t_string = unk_t_string + to_string(counter); }

    cout << "Identified the generic unknown type as " << unk_t_string << ". Adding it to the alphabet." << endl;
    id.add_type(unk_t_string);
}