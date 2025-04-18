/**
 * Hielke Walinga
 */

#ifndef _SQLDB_SUL_REGEX_ORACLE_H_
#define _SQLDB_SUL_REGEX_ORACLE_H_

#include "base_oracle.h"
#include "misc/sqldb.h"
//#include "sqldb_sul.h"
#include <optional>
#include <utility>

class sqldb_sul_regex_oracle : public base_oracle {
  private:
    int parts = 1; // Amount of parts to split the regex in.
  protected:
    // TODO: need to redefine sul again?
    std::shared_ptr<sul_base> sul;
    std::shared_ptr<sqldb_sul> my_sqldb_sul;

  public:
    explicit sqldb_sul_regex_oracle(std::unique_ptr<sul_base>& sul) : base_oracle(sul) {
        my_sqldb_sul = dynamic_pointer_cast<sqldb_sul>(sul);
        if (my_sqldb_sul == nullptr) {
            throw std::logic_error("sqldb_sul_regex_oracle only works with sqldb_sul.");
        }
    };

    std::optional<psql::record> equivalence_query_db(state_merger* merger);

    std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger);
};

#endif
