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