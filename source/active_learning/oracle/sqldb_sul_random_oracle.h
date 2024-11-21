/**
 * Hielke Walinga
 */

#ifndef _SQLDB_SUL_RANDOM_ORACLE_H_
#define _SQLDB_SUL_RANDOM_ORACLE_H_

#include "oracle_base.h"
#include "misc/sqldb.h"
//#include "sqldb_sul.h"
#include <optional>
#include <utility>

class sqldb_sul_random_oracle : public oracle_base {
  protected:
    // TODO: need to redefine sul again?
    std::shared_ptr<sul_base> sul;
    std::shared_ptr<sqldb_sul> my_sqldb_sul;

    virtual void reset_sul() override{};

  public:
    explicit sqldb_sul_random_oracle(std::unique_ptr<sul_base>& sul) : oracle_base(sul) {
        my_sqldb_sul = dynamic_pointer_cast<sqldb_sul>(sul);
        if (my_sqldb_sul == nullptr) {
            throw std::logic_error("sqldb_sul_random_oracle only works with sqldb_sul.");
        }
    };

    std::optional<psql::record> equivalence_query_db(state_merger* merger,
                                                     const std::unordered_set<int>& added_traces);

    std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger);
};

#endif
