/**
 * @file running_mode_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "running_mode_base.h"
#include "parameters.h"
#include "inputdatalocator.h"

#include <memory>

using namespace std;

/**
 * @brief Reads the entire input file all at once and stores it in input data.
 */
void running_mode_base::read_input_file() {
    ifstream input_stream(INPUT_FILE);

    if(!input_stream) {
        LOG_S(ERROR) << "Input file not found, aborting";
        cerr << "Input file not found, aborting" << endl;
        exit(-1);
    } else {
        cout << "Using input file: " << INPUT_FILE << endl;
    }

    bool read_csv = false;
    if(INPUT_FILE.ends_with(".csv")){
        read_csv = true;
    }

    if(read_csv) {
        parser = make_unique<csv_parser>(input_stream, csv::CSVFormat().trim({' '}));
    } else {
        parser = make_unique<abbadingoparser>(input_stream);
    }

    if (SLIDING_WINDOW) {
        id.read_slidingwindow(parser.get(),
                               SLIDING_WINDOW_SIZE,
                               SLIDING_WINDOW_STRIDE,
                               SLIDING_WINDOW_TYPE);
    } else {
        id.read(parser.get());
    }
}

/**
 * @brief For convenience. Nearly all modes execute this code.
 * 
 */
void running_mode_base::initialize(){
  // Initialize main classes.
  inputdata_locator::provide(&id);
  apta* the_apta = new apta();
  evaluation_function *eval = get_evaluation();
  state_merger* merger = new state_merger(&id, eval, the_apta);

  the_apta->set_context(merger);
  eval->set_context(merger);
}