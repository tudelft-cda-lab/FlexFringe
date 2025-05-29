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

#ifndef _SQLDB_SUL_H_
#define _SQLDB_SUL_H_

#include "misc/sqldb.h"
#include "sul_base.h"
#include <optional>
#include <utility>
#include <vector>

class sqldb_sul : public sul_base {
  public:
    psql::db& my_sqldb;

    void reset() override{};
    explicit sqldb_sul(psql::db& db);
    virtual void pre(inputdata& id) override;

    const std::unordered_set<int>& added_traces;

    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;
    const sul_response regex_equivalence(const std::string& regex, int type);
    const std::vector<sul_response> prefix_query(const std::vector<int>& prefix, int n);
};

#endif
