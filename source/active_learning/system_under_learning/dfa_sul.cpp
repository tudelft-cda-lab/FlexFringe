/**
 * @file dfa_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-04-06
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "dfa_sul.h"
#include "benchmark_dfaparser.h"
#include "common_functions.h"
#include "parameters.h"

#include <fstream>
#include <stdexcept>

using namespace std;
using namespace active_learning_namespace;

const bool DEBUG = true;

/**
 * @brief This function does two tests: First it checks if there's an APTA file, and if yes reads it.
 * If this check fails it checks if the input file is json formatted, and if yes reads that one. Otherwise
 * throw an exception.
 *
 * The reason for the design is that we can have multiple modes, either only active learning (then we
 * use the INPUT file), or active and passive learning combined (then we use the APTA file).
 *
 * @param id The input data (not used here).
 */
void dfa_sul::pre(inputdata& id) {
    ifstream input_apta_stream;

    // read_json stores alphabet and types in inputdata
    if (!APTA_FILE.empty()) {
        input_apta_stream = ifstream(APTA_FILE);
        cout << "Reading apta file (SUT) - " << APTA_FILE << endl;
        sut->read_json(input_apta_stream);
    } else if (INPUT_FILE.compare(INPUT_FILE.length() - 5, INPUT_FILE.length(), ".json") == 0) {
        input_apta_stream = ifstream(INPUT_FILE);
        cout << "Reading input file (SUT) - " << INPUT_FILE << endl;
        sut->read_json(input_apta_stream);
    } else if (INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".dot") == 0) {
        // TODO: throw in a benchmark parser
        auto input_dfa_stream = ifstream(INPUT_FILE);
        auto parser = benchmark_dfaparser();
        sut = parser.read_input(input_dfa_stream);

        if (DEBUG) {
            stringstream dot_output_buf;
            sut->print_dot(dot_output_buf);
            string dot_output = "// produced with flexfringe // " + COMMAND + '\n' + dot_output_buf.str();

            ofstream output1("read_machine.dot");
            if (output1.fail()) {
                throw std::ofstream::failure("Unable to open file for writing.");
            }
            output1 << dot_output;
            output1.close();
        }
    } else {
        throw logic_error("Problem with reading input");
    }
}

const sul_response dfa_sul::do_query(const std::vector<int>& query_trace, inputdata& id) const {
    // TODO: query/predict the apta, return the type as predicted
    trace* tr = mem_store::create_trace(&id);
    add_sequence_to_trace(tr, query_trace);

    const int pred_type = predict_type_from_trace(tr, sut.get(), id);
    return sul_response(pred_type);
}