/**
 * @file algorithm_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "algorithm_base.h"
#include "parameters.h"

#include "abbadingoparser.h"
#include "csvparser.h"
#include "input/abbadingoreader.h"
#include "inputdata.h"
#include "inputdatalocator.h"

using namespace std;

ifstream algorithm_base::get_inputstream() const {
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

unique_ptr<parser> algorithm_base::get_parser(ifstream& input_stream) const {
    if (INPUT_FILE.ends_with(".csv")) {
        return make_unique<csv_parser>(input_stream, csv::CSVFormat().trim({' '}));
    } else {
        return make_unique<abbadingoparser>(input_stream);
    }
}

inputdata* algorithm_base::get_inputdata() const {
    ifstream input_stream = get_inputstream();

    inputdata* id = inputdata_locator::get();

    auto input_parser = get_parser(input_stream);
    id->read(input_parser.get());
    input_stream.close();
    return id;
}

/**
 * @brief A standard way to initialize the oracle, since most of the algorithms use this method.
 * 
 */
void algorithm_base::init_standard() {
    auto sul = sul_factory::create_sul(AL_SYSTEM_UNDER_LEARNING);
    auto ds_handler = ds_handler_factory::create_ds_handler(sul, AL_II_NAME);
    this->oracle = oracle_factory::create_oracle(sul, AL_ORACLE, ds_handler);
}

/**
 * @brief Sets the internal types representation. We need this for classification SULs (hence algorithms that
 * output classification state machines like DFA).
 * 
 * IMPORTANT: Must be called AFTER oracle/sul have been initialized completely, and after inputdata
 * has been initialized completely.
 * 
 */
void algorithm_base::set_types() {
    const auto types = oracle->get_types();
    if(types.size()==0){
        cout << "Empty set of types received. Skipping type initialization." << endl;
        return;
    }

    auto* id = inputdata_locator::get();
    id->set_types(types);
}