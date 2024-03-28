/**
 * Hielke Walinga
 */

#ifndef _SQLDB_SUL_REGEX_ORACLE_H_
#define _SQLDB_SUL_REGEX_ORACLE_H_

#include "eq_oracle_base.h"
#include "misc/sqldb.h"
#include "sqldb_sul.h"
#include <optional>
#include <utility>

class sqldb_sul_regex_oracle : public eq_oracle_base {
  private:
    int parts = 1; // Amount of parts to split the regex in.
  protected:
    std::shared_ptr<sul_base> sul;
    std::shared_ptr<sqldb_sul> my_sqldb_sul;
    std::unique_ptr<search_base> search_strategy;

    virtual void reset_sul() override{};

  public:
    explicit sqldb_sul_regex_oracle(std::shared_ptr<sul_base>& sul) : eq_oracle_base(sul) {
        my_sqldb_sul = dynamic_pointer_cast<sqldb_sul>(sul);
        if (my_sqldb_sul == nullptr) {
            throw logic_error("sqldb_sul_regex_oracle only works with sqldb_sul.");
        }
    };

    std::optional<psql::record> equivalence_query_db(state_merger* merger,
                                                     [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher);

    optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher);
};

#endif
