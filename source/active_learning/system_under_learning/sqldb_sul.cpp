/**
 * @file sqldb_sul.cpp
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief Present the SUL as from a SQL database
 * @version 0.1
 * @date 2023-15-12
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "sqldb_sul.h"

#include "loguru.hpp"

sqldb_sul::sqldb_sul(sqldb& db) : my_sqldb(db) {}

void sqldb_sul::pre(inputdata& id) {
    /* LOG_S(INFO) << fmt::format("Inferring alphabet from {0}_meta table.", my_sqldb->table_name); */
    // provide the alphabet to inputdata
    // TODO
}

bool sqldb_sul::is_member(const std::vector<int>& query_trace) const {
    return true;
    // convert vector of ints to string given the alphabet
    // TODO
}

const int sqldb_sul::query_trace(const std::vector<int>& query_trace, inputdata& id) const {
    return 1;
    // convert vector of ints to string given the alphabet
    // TODO
}
