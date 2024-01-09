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

class sqldb_sul : public sul_base {
  private:
    sqldb& my_sqldb;

  protected:
    void reset() override{};
    const double get_string_probability(const std::vector<int>& query_trace, inputdata& id) const override{};

    bool is_member(const std::vector<int>& query_trace) const override;
    const int query_trace(const std::vector<int>& query_trace, inputdata& id) const override;

  public:
    explicit sqldb_sul(sqldb& db);
    virtual void pre(inputdata& id) override;
    const int query_trace_maybe(const std::vector<int>& query_trace, inputdata& id) const override;
    sqldb& get_sqldb() override { return my_sqldb; };
};

#endif
