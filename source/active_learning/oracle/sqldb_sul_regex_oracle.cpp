/**
 * Hielke Walinga
 */

#include "sqldb_sul_regex_oracle.h"
#include "common.h"
#include "input/inputdatalocator.h"
#include "misc/printutil.h"
#include "misc/sqldb.h"
#include "misc/trim.h"
#include "parameters.h"
#include "regex_builder.h"
#include "utility/loguru.hpp"
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

std::optional<std::pair<std::vector<int>, sul_response>>
sqldb_sul_regex_oracle::equivalence_query(state_merger* merger) {
    inputdata& id = *(merger->get_dat());
    apta& hypothesis = *(merger->get_aut());

    static const auto& types = id.get_types();

    auto my_types = types;
    std::shuffle(std::begin(my_types), std::end(my_types), RNG);
    std::cout << "Types shuffled: " << my_types << std::endl;

    auto coloring = std::make_tuple(PRINT_RED, PRINT_BLUE, PRINT_WHITE);
    regex_builder builder = regex_builder(hypothesis, *merger, coloring, psql::db::num2str);

    int i = 0;
    std::vector<bool> errors(types.size(), false);
    int parts = 1;
    while (i < my_types.size()) {
        int type = my_types[i];
        int nodes = builder.get_types_map()[inputdata_locator::get()->string_from_type(type)].size();

        if (nodes > 30) {
            LOG_S(INFO) << "Stop regex, is likely too complex.";
            throw std::runtime_error("Regex likely too complex.");
        }

        for (std::string regex : builder.to_regex(type, parts)) {
            if (regex.size() > 1000000000) {
                LOG_S(INFO) << "Stop regex, is likely too complex.";
                throw std::runtime_error("Regex likely too complex.");
            }
            std::cout << "We have " << parts << " parts and got a regex with size " << regex.size() << std::endl;
            if (regex.size() > 400000) { // too complex
                errors[i] = true;
                continue;
            }
/* When pqxx is used, it needs to be guarded out since we want to compile on platforms without libpq and pqxx still. */
#ifdef __FLEXFRINGE_DATABASE
            try {
                sul_response cex = my_sqldb_sul->regex_equivalence(regex, type);
                if (cex.has_int_val()) {
                    // Found counter example, return immidiatly.
                    return std::make_optional<std::pair<std::vector<int>, sul_response>>(
                        std::make_pair(cex.GET_INT_VEC(), cex));
                }
            } catch (const pqxx::data_exception& e) {
                // Regex got too big, lets split into multiple parts.
                errors[i] = true;
                std::string exc{e.what()};
                trim(exc);
                LOG_S(INFO) << "We got error: " << exc;
                if (exc.find("too complex") != std::string::npos) {
                    continue;
                } else {
                    throw std::runtime_error("Something wrong with regex to database: " + exc);
                }
            }
#endif /* __FLEXFRINGE_DATABASE */
        }
        if (errors[i]) {
            if (parts == nodes) {
                // Looked at all nodes individually, go to new type.
                errors[i] = true;
                i++;
                parts = 1;
                continue; // Let's investigate next type.
            }
            // No new type, but try this type again with bigger parts number (thus larger split).
            parts *= 2;
            if (parts > nodes)
                parts = nodes; // final try -> all nodes individually.

            errors[i] = false;
            LOG_S(INFO) << "Increase the amount of parts we split the regexes with " << parts;
            continue;
        }
        // No counter example yet, check new type.
        i++;
        parts = 1;
    }

    // final procedure.

    for (bool e : errors) {
        if (e) {
            throw std::runtime_error(
                "Regex was too complex, no counter example found, this is your final state machine.");
        }
    }

    // No difference found: No counter example: Found the truth.
    return std::nullopt;
}
