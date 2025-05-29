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
#include "sul_base.h"
#include "utility/loguru.hpp"

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

const sul_response sqldb_sul::do_query(const std::vector<int>& query_trace, inputdata& id) const {
    auto rec = my_sqldb.query_trace_opt(my_sqldb.vec2str(query_trace));
    return sul_response(rec.type, rec.pk, rec.trace)
}

const sul_response sqldb_sul::regex_equivalence(const std::string& regex, int type) {
    auto rec = my_sqldb.regex_equivalence(regex, type);
    return sul_response(rec.type, rec.pk, rec.trace)
}

const std::vector<sul_response> sqldb_sul::prefix_query(const std::vector<int>& prefix, int n) {
    auto rec = my_sqldb.prefix_query(my_sqldb.vec2str(prefix), n);
    return sul_response(rec.type, rec.pk, rec.trace)
}
