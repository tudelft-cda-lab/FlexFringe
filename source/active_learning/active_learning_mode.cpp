/**
 * @file active_learning_mode.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The main subroutine that is starting the active learning.
 * @version 0.1
 * @date 2023-03-08
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "active_learning_mode.h"
#include "algorithm_base.h" 
#include "factories.h"

#include "inputdatalocator.h"
#include "misc/printutil.h"
#include "utility/loguru.hpp"

#include <cassert>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace std;

void active_learning_mode::initialize(){
    if(OUTPUT_FILE.empty()) {
        if (!POSTGRESQL_TBLNAME.empty()) {
            OUTPUT_FILE = POSTGRESQL_TBLNAME + ".ff";
        } else {
            OUTPUT_FILE = INPUT_FILE + ".ff";
        }
    }
}

/* ifstream active_learning_main_func::get_inputstream() const {
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
} */

/**
 * @brief Selects the parameters the algorithm runs with and runs the algorithm.
 *
 */
int active_learning_mode::run() {
    inputdata id;
    inputdata_locator::provide(&id);
    if(ENSEMBLE_RUNS <= 0) // TODO: delete this one?
        throw logic_error("ensemble runs must be larger then 0");

    unique_ptr<algorithm_base> algorithm = algorithm_factory::create_algorithm_obj();
    algorithm->run(id);

        // Hielke: Can we we this one better? For example, we do it in the constructor of the corresponding algorithms
/*         LOG_S(INFO) << "Learning (partly) passively. Therefore read in input-data.";
        get_inputdata();

        sul->pre(id);
        algorithm->run(id); */
    return EXIT_SUCCESS;
}
