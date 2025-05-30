/**
 * @file sul_factory.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2025-05-10
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "sul_factory.h"
#include "inputdatalocator.h"

// the SULs
#include "database_sul.h"
#include "dfa_sul.h"
#include "input_file_sul.h"
#include "sqldb_sul.h"

// the neural network SULs
#include "nn_binary_output_sul.h"
#include "nn_discrete_and_float_output_sul.h"
#include "nn_discrete_output_and_hidden_reps_sul.h"
#include "nn_discrete_output_sul.h"
#include "nn_float_output_sul.h"
#include "nn_float_vector_output_sul.h"

using namespace std;

/**
 * @brief Gets the SUl, initializes it, and returns it.
 */
shared_ptr<sul_base> sul_factory::create_sul(string_view sul_name) {
    shared_ptr<sul_base> res;
    // non-neural network suls
    if (sul_name == "input_file_sul")
        res = make_shared<input_file_sul>();
    else if (sul_name == "dfa_sul")
        res = make_shared<dfa_sul>();
    else if (sul_name == "database_sul")
        res = make_shared<database_sul>();
    else if (sul_name == "sqldb_sul") {
        auto my_sqldb = make_shared<psql::db>(POSTGRESQL_TBLNAME, POSTGRESQL_CONNSTRING);

        std::cout << "GET" << std::endl;
        my_sqldb->get_alphabet();
        std::cout << "GOT" << std::endl;

        res = make_shared<sqldb_sul>(*my_sqldb);
    }

    // the neural network suls
    else if (sul_name == "nn_binary_output_sul")
        res = make_shared<nn_binary_output_sul>();
    else if (sul_name == "nn_discrete_and_float_output_sul")
        res = make_shared<nn_discrete_and_float_output_sul>();
    else if (sul_name == "nn_discrete_output_and_hidden_reps_sul")
        res = make_shared<nn_discrete_output_and_hidden_reps_sul>();
    else if (sul_name == "nn_discrete_output_sul")
        res = make_shared<nn_discrete_output_sul>();
    else if (sul_name == "nn_float_output_sul")
        res = make_shared<nn_float_output_sul>();
    else if (sul_name == "nn_float_vector_output_sul")
        res = make_shared<nn_float_vector_output_sul>();
    else
        throw invalid_argument(
            "Input parameter specifying system under learning has been invalid. Please check your input.");

    inputdata* id = inputdata_locator::get();
    if (id == nullptr)
        throw logic_error("Inputdata must exist the moment the SUL is created");

    res->pre(*id);

    return res;
}
