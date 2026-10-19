/**
 * @file regex_mode.cpp
 * @author Hielke Walinga
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "load_sqldb_mode.h"
#include "inputdatalocator.h" 

#include <iostream>
#include <ranges>
#include "misc/sqldb.h"

void load_sqldb_mode::initialize(){
    // do nothing
}

int load_sqldb_mode::run(){
    std::ifstream input_stream = get_inputstream();
    std::cout << "Selected to use the SQL database. Creating new inputdata object and loading traces.";
    abbadingo_inputdata id;
    inputdata_locator::provide(&id);
    auto my_sqldb = make_unique<psql::db>(POSTGRESQL_TBLNAME, POSTGRESQL_CONNSTRING);
    my_sqldb->load_traces(id, input_stream);

    return EXIT_SUCCESS;
}

void load_sqldb_mode::generate_output(){
    // do nothing
}
