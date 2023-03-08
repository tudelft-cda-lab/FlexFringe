/**
 * @file testinputfilesul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-03-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "input_file_sul.h"
#include "inputdata.h"
#include "abbadingoparser.h"
#include "inputdatalocator.h"
#include "csvparser.h"
#include "parameters.h"

#include "catch.hpp"
#include "definitions.h"

#include <vector>

using namespace std;
using namespace active_learning_namespace;

//using Catch::Matchers::Equals;

TEST_CASE("Input file SUL", "[SUL]") {

    // with this one runtest must be executed from flexfringe root directory
    INPUT_FILE = "./data/PAutomaC-competition_sets/5.pautomac.train.dat";
    input_file_sul sul;
    inputdata id;
    sul.parse_input(id);

    SECTION("Check initialization"){
        CHECK(true); // TODO
    }
}