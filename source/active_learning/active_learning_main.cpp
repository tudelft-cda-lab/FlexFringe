/**
 * @file active_learning_main.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The main subroutine that is starting the active learning.
 * @version 0.1
 * @date 2023-03-08
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "active_learning_main.h"

#include "misc/printutil.h"
#include "utility/loguru.hpp"

#include <cassert>
#include <fstream>
#include <stdexcept>
#include <string>

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace std;
using namespace active_learning_namespace;

ifstream active_learning_main_func::get_inputstream() const {
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

unique_ptr<parser> active_learning_main_func::get_parser(ifstream& input_stream) const {
    if (INPUT_FILE.ends_with(".csv")) {
        return make_unique<csv_parser>(input_stream, csv::CSVFormat().trim({' '}));
    } else {
        return make_unique<abbadingoparser>(input_stream);
    }
}

inputdata* active_learning_main_func::get_inputdata() const {
    ifstream input_stream = get_inputstream();

    inputdata* id = inputdata_locator::get();

    auto input_parser = get_parser(input_stream);
    id->read(input_parser.get());
    input_stream.close();
    return id;
}

/**
 * @brief Selects the SUL to be used.
 *
 * @return shared_ptr<sul_base> The sul.
 */
unique_ptr<sul_base> active_learning_main_func::select_sul_class(const bool ACTIVE_SUL) const {

}

/**
 * @brief Selects the oracle to be used. In case alternative oracles want to be written.
 *
 * @return unique_ptr<oracle_base> The oracle.
 */
unique_ptr<oracle_base> active_learning_main_func::select_oracle_class(unique_ptr<sul_base>&& sul,
                                                                          const bool ACTIVE_SUL) const {

}

/**
 * @brief Selects the parameters the algorithm runs with and runs the algorithm.
 *
 */
void active_learning_main_func::run_active_learning() {
    inputdata id;
    inputdata_locator::provide(&id);
    assertm(ENSEMBLE_RUNS > 0, "nruns parameter must be larger than 0 for active learning.");

    unique_ptr<sul_base> sul = select_sul_class(ACTIVE_SUL);
    unique_ptr<oracle_base> oracle = select_oracle_class(move(sul), ACTIVE_SUL);

    unique_ptr<algorithm_base> algorithm;
    if (ACTIVE_LEARNING_ALGORITHM == "l_star") {
        algorithm = make_unique<lstar_algorithm>(move(oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "l_star_imat") {
        STORE_ACCESS_STRINGS = true;
        algorithm = make_unique<lstar_imat_algorithm>(move(oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "l_sharp") {
        STORE_ACCESS_STRINGS = true;
        algorithm = make_unique<lsharp_algorithm>(move(oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "p_l_sharp") {
        STORE_ACCESS_STRINGS = true;
        algorithm = make_unique<probabilistic_lsharp_algorithm>(move(oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "weighted_l_sharp") {
        STORE_ACCESS_STRINGS = true;
        algorithm = make_unique<weighted_lsharp_algorithm>(move(oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "l_dot") {
        STORE_ACCESS_STRINGS = true; // refinement uses this to get nodes, but that seems buggy somehow.
        //algorithm = make_unique<ldot_algorithm>(move(oracle));
        cerr << "ldot currently commented out. Terminating" << endl;
        exit(0);
    } else if (ACTIVE_LEARNING_ALGORITHM == "paul") {
        STORE_ACCESS_STRINGS = true;
        algorithm = make_unique<paul_algorithm>(move(oracle));
    } else {
        throw logic_error("Fatal error: Unknown active_learning_algorithm flag used: " + ACTIVE_LEARNING_ALGORITHM);
    }

    if (ACTIVE_SUL && ACTIVE_LEARNING_ALGORITHM != "paul") {
        LOG_S(INFO) << "We do not want to run the input file, alphabet and input data must be inferred from SUL.";

        sul->pre(id);
        algorithm->run(id);
    } else {
        LOG_S(INFO) << "Learning (partly) passively. Therefore read in input-data.";
        get_inputdata();

        sul->pre(id);
        algorithm->run(id);
    }
}
