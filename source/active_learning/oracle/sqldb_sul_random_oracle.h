/**
 * Hielke Walinga
 */

#ifndef _SQLDB_SUL_RANDOM_ORACLE_H_
#define _SQLDB_SUL_RANDOM_ORACLE_H_

#include "base_oracle.h"
#include "misc/sqldb.h"
#include "sqldb_sul.h"
#include <optional>
#include <utility>

class sqldb_sul_random_oracle : public base_oracle {
  protected:
    std::shared_ptr<sul_base> sul;
    // Need a sqldb_sul for specific queries only available on the sqldb_sul.
    std::shared_ptr<sqldb_sul> my_sqldb_sul;

  public:
    explicit sqldb_sul_random_oracle(const std::shared_ptr<sul_base>& sul) : base_oracle(sul) {
        my_sqldb_sul = std::dynamic_pointer_cast<sqldb_sul>(sul);
        if (my_sqldb_sul == nullptr) {
            throw std::logic_error("sqldb_sul_random_oracle only works with sqldb_sul.");
        }
    };

    std::optional<std::pair<std::vector<int>, sul_response>> equivalence_query(state_merger* merger);
};

#endif
