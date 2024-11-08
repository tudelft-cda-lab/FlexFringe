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
#include "misc/printutil.h"
#include "misc/sqldb.h"
#include "utility/loguru.hpp"
#include <sstream>

sqldb_sul::sqldb_sul(psql::db& db) : my_sqldb(db) {}

void sqldb_sul::pre(inputdata& id) {
    LOG_S(INFO) << fmt::format("Inferring alphabet from {0}_meta table.", my_sqldb.get_table_name());

    auto alphabet = my_sqldb.get_alphabet();
    std::map<std::string, int> r_alphabet;
    for (int i = 0; i < alphabet.size(); i++) {
        std::string symbol = alphabet[i];
        r_alphabet[symbol] = i;
    }
    id.set_alphabet(r_alphabet);

    auto types = my_sqldb.get_types();
    std::map<std::string, int> r_types;
    for (int i = 0; i < types.size(); i++) {
        std::string symbol = types[i];
        r_types[symbol] = i;
    }
    id.set_types(r_types);

    LOG_S(INFO) << "Set the alphabet and types to inputdata.";
}

bool sqldb_sul::is_member(const std::vector<int>& query_trace) const {
    return my_sqldb.is_member(my_sqldb.vec2str(query_trace));
}

const int sqldb_sul::query_trace(const std::vector<int>& query_trace, inputdata& id) const {
    return my_sqldb.query_trace(my_sqldb.vec2str(query_trace));
}

const int sqldb_sul::query_trace_maybe(const std::vector<int>& query_trace, inputdata& id) const {
    return my_sqldb.query_trace_maybe(my_sqldb.vec2str(query_trace));
}

const std::optional<psql::record> sqldb_sul::query_trace_opt(const std::vector<int>& query_trace) const {
    return my_sqldb.query_trace_opt(my_sqldb.vec2str(query_trace));
}
std::optional<psql::record> sqldb_sul::regex_equivalence(const std::string& regex, int type) {
    return my_sqldb.regex_equivalence(regex, type);
}

std::vector<psql::record> sqldb_sul::prefix_query(const std::vector<int>& prefix, int n) {
    return my_sqldb.prefix_query(my_sqldb.vec2str(prefix), n);
}
