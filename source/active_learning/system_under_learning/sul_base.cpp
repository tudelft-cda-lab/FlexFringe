/**
 * @file sul_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Implementation of SUL base class.
 * @version 0.1
 * @date 2023-02-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "sul_base.h"
#include "parameters.h"

#include <iostream>
#include <stdexcept>

using namespace std;

ifstream sul_base::get_input_stream() const {
    ifstream input_stream(INPUT_FILE);
    cout << "Input file: " << INPUT_FILE << endl;

    if (!input_stream) {
        cerr << "Input file not found, aborting" << endl;
        exit(-1);
    } else {
        cout << "Using input file: " << INPUT_FILE << endl;
    }
    return input_stream;
}

const double sul_base::get_string_probability(const std::vector<int>& query_trace, inputdata& id) const {
    throw logic_error("The SUL tries to infer string probability, but this SUL does not support this. \
  Please change the program settings. Aborting program.");
}

const vector<float> sul_base::get_weight_distribution(const std::vector<int>& query_trace, inputdata& id) const {
    throw logic_error(
        "This SUL does not support inference of the weight distribution. Please change the program settings. \
  Aborting program.");
};
