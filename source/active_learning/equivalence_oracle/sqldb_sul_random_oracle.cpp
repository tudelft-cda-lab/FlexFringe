/**
 * Hielke Walinga
 */

#include "sqldb_sul_random_oracle.h"
#include "common_functions.h"
#include "inputdata.h"
#include "misc/printutil.h"
#include "misc/sqldb.h"
#include "utility/loguru.hpp"
#include <optional>
#include <parameters.h>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

std::optional<psql::record> sqldb_sul_random_oracle::equivalence_query_db(state_merger* merger,
                                                                          const std::unique_ptr<base_teacher>& teacher,
                                                                          unordered_set<int> added_traces) {
    inputdata& id = *(merger->get_dat());
    int max_pk = my_sqldb_sul->my_sqldb.max_trace_pk();
    std::vector<int> possible_ids;

    for (int pk = 0; pk <= max_pk; pk++) {
        if (!added_traces.contains(pk))
            possible_ids.push_back(pk);
    }
    std::shuffle(possible_ids.begin(), possible_ids.end(), RNG);

    for (int i : possible_ids) {
        auto rec_maybe = my_sqldb_sul->my_sqldb.select_by_pk(i);
        if (!rec_maybe)
            continue;
        psql::record rec = rec_maybe.value();

        trace* t = active_learning_namespace::vector_to_trace(rec.trace, id, rec.type);
        apta_node* ending_state = merger->get_state_from_trace(t)->find();
        tail* ending_tail = t->end_tail;
        if (ending_tail->is_final())
            ending_tail = ending_tail->past();

        // predict the type from the current structure.
        int type = ending_state->get_data()->predict_type(ending_tail);

        // if different from db, return
        if (type != rec.type) {
            return make_optional<psql::record>(rec);
        }
    }

    // No difference found: No counter example: Found the truth.
    return std::nullopt;
}

optional<std::pair<std::vector<int>, int>>
sqldb_sul_random_oracle::equivalence_query(state_merger* merger, const std::unique_ptr<base_teacher>& teacher) {
    unordered_set<int> empty_set;
    optional<psql::record> r_maybe = equivalence_query_db(merger, teacher, empty_set);
    if (!r_maybe)
        return std::nullopt;
    psql::record r = r_maybe.value();
    return std::make_optional<std::pair<std::vector<int>, int>>(std::make_pair(r.trace, r.type));
}
