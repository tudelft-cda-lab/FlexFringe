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
  protected:
    void reset() override{};
    const double get_string_probability(const std::vector<int>& query_trace, inputdata& id) const override{};

    bool is_member(const std::vector<int>& query_trace) const override;
    const int query_trace(const std::vector<int>& query_trace, inputdata& id) const override;

  public:
    psql::db& my_sqldb;
    explicit sqldb_sul(psql::db& db);
    virtual void pre(inputdata& id) override;
    const int query_trace_maybe(const std::vector<int>& query_trace) const;
    const std::optional<psql::record> query_trace_opt(const std::vector<int>& query_trace) const;
    psql::db& get_sqldb() { return my_sqldb; };
    std::optional<psql::record> regex_equivalence(const std::string& regex, int type);
    std::vector<psql::record> prefix_query(const std::vector<int>& prefix, int n);
};

#endif
