/**
 * Hielke Walinga
 */

#include "sqldb_sul_regex_oracle.h"
#include "misc/sqldb.h"
#include "regex_builder.h"
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using CEX = std::pair<std::vector<int>, int>; // alphabet vector with its type as the counter example.

std::optional<CEX> sqldb_sul_regex_oracle::equivalence_query(state_merger* merger,
                                                             const std::unique_ptr<base_teacher>& teacher) {
    inputdata& id = *(merger->get_dat());
    apta& hypothesis = *(merger->get_aut());
    static const auto& types = id.get_types();

    auto coloring = std::make_tuple(PRINT_RED, PRINT_BLUE, PRINT_WHITE);
    regex_builder builder = regex_builder(hypothesis, *merger, coloring, sqldb::num2str);

    // TODO; perform alteration for the types to check as a performance enhancement.
    for (auto type : types) {
        std::optional<CEX> cex = my_sqldb_sul->regex_equivalence(builder.to_regex(type), type);
        if (cex)
            return cex;
    }

    // No difference found: No counter example: Found the truth.
    return std::nullopt;
}
