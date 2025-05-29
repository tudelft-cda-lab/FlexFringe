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

// true random makes use of id's that are shuffled. That is a true random oracle,
// however, using db::stream_traces (in a fixed order) is faster.
const bool TRUE_RANDOM = false;

std::optional<std::pair<std::vector<int>, sul_response>>
sqldb_sul_random_oracle::equivalence_query(state_merger* merger) {
    inputdata& id = *(merger->get_dat());
    int n = 0;
    const int max_pk = my_sqldb_sul->my_sqldb.max_trace_pk();

    if (TRUE_RANDOM) {
        // Based on a shuffled list of id's

        std::vector<int> possible_ids;

        for (int pk = 0; pk <= max_pk; pk++) {
            if (!my_sqldb_sul.added_traces.contains(pk))
                possible_ids.push_back(pk);
        }
        std::shuffle(possible_ids.begin(), possible_ids.end(), RNG);

        for (const int i : possible_ids) {
            auto rec_maybe = my_sqldb_sul->my_sqldb.select_by_pk(i);
            if (!rec_maybe)
                continue;
            psql::record rec = rec_maybe.value();

            trace* t = active_learning_namespace::vector_to_trace(rec.trace, id, rec.type);
            apta_node* ending_state = merger->get_state_from_trace(t);

            // Found a path in database that is not in the APTA
            if (ending_state == nullptr)
                return std::make_optional<psql::record>(rec);

            ending_state = ending_state->find();
            tail* ending_tail = t->end_tail;

            if (ending_tail->is_final())
                ending_tail = ending_tail->past();

            // predict the type from the current structure.
            const int type = ending_state->get_data()->predict_type(ending_tail);
            t->erase();

            if (n++ % 10000 == 0)
                LOG_S(INFO) << "Looked at " << n << "/" << possible_ids.size() << " queries for equivalence.";

            // if different from db, return
            if (type != rec.type) {
                auto sul_resp = sul_response(rec.type, rec.pk, rec.trace);
                return std::make_optional<std::pair<std::vector<int>, int>>(std::make_pair(rec.trace, sul_resp));
            }
        }

        // No difference found: No counter example: Found the truth.
        return std::nullopt;
    } else {
        // based on a fixed stream of db::stream_traces

        std::optional<psql::record> ans = std::nullopt;
        const auto g = [&id, &merger, &ans, &n, &max_pk](psql::record rec) {
            trace* t = active_learning_namespace::vector_to_trace(rec.trace, id, rec.type);
            apta_node* ending_state = merger->get_state_from_trace(t);

            // Found a path in database that is not in the APTA
            if (ending_state == nullptr) {
                ans.emplace(rec);
                return false;
            }

            ending_state = ending_state->find();
            tail* ending_tail = t->end_tail;

            if (ending_tail->is_final())
                ending_tail = ending_tail->past();

            // predict the type from the current structure.
            const int type = ending_state->get_data()->predict_type(ending_tail);
            t->erase();

            if (n++ % 10000 == 0)
                LOG_S(INFO) << "Looked at " << n << "/" << max_pk << " queries for equivalence.";

            // if different from db, return
            if (type != rec.type) {
                ans.emplace(rec);
                return false; // stop streaming
            }
            return true;
        };

        my_sqldb_sul->my_sqldb.stream_traces(g);

        // Either lambda g sets ans to a value or it remain the nullopt.

        if (!ans)
            return std::nullopt;
        auto rec = ans.value();
        auto sul_resp = sul_response(rec.type, rec.pk, rec.trace);
        return std::make_optional<std::pair<std::vector<int>, int>>(std::make_pair(rec.trace, sul_resp));
    }
}
