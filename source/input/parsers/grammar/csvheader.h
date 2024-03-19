//
// Created by tom on 1/24/23.
//

#ifndef FLEXFRINGE_CSVHEADER_H
#define FLEXFRINGE_CSVHEADER_H

#include <lexy/dsl/literal.hpp>
#include <lexy/action/parse.hpp> // lexy::parse
#include <lexy/callback.hpp>     // value callbacks
#include <lexy/dsl.hpp>          // lexy::dsl::*

#include <string>
#include <utility>
#include <vector>
#include <set>
#include <iostream>
#include <optional>

struct col_name_info {
    std::optional<std::string> type_name;
    std::optional<std::set<std::string>> attr_types;
    std::string name;
};

namespace {
    namespace csv_header_grammar {
        namespace dsl = lexy::dsl;

        struct number {
            static constexpr auto rule = dsl::integer<std::uint64_t>;
            static constexpr auto value = lexy::as_integer<std::uint64_t>;
        };

        struct name {

            struct name_error {
                static constexpr auto name = "Incomplete attribute specification (did you add types?)";
            };

            static constexpr auto rule = [] {
                auto word = dsl::identifier(dsl::ascii::word);
                auto peek = dsl::peek(LEXY_LIT("tattr") | LEXY_LIT("attr"));
                return peek >> dsl::error<name_error> | dsl::else_ >> word;
            }();
            static constexpr auto value = lexy::as_string<std::string>;
        };

        struct attr_name {
            static constexpr auto rule = dsl::capture(LEXY_LIT("tattr")) | dsl::capture(LEXY_LIT("attr"));
            static constexpr auto value = lexy::as_string<std::string>;
        };

        // Attribute types: set of d, s, f, t
        struct attr_types {
            static constexpr auto rule = [] {
                auto discrete = LEXY_LIT("d");
                auto splittable = LEXY_LIT("s");
                auto distributionable = LEXY_LIT("f");
                auto target = LEXY_LIT("t");

                auto item = dsl::capture(dsl::literal_set(
                        discrete, splittable, distributionable, target
                ));

                return dsl::list(item);
            }();

            static constexpr auto value = lexy::fold_inplace<std::set<std::string>>(
                    std::initializer_list<std::string> {},
                    [](std::set<std::string>& acc, const auto& val) {
                       acc.insert(lexy::as_string<std::string>(val));
                    });
        };

        struct col_name {
            static constexpr auto rule = [] {
                auto just_name = dsl::p<name>;
                auto type_w_name = dsl::p<name> + LEXY_LIT(":") + dsl::p<name>;
                auto attr_w_name = dsl::p<attr_name> + LEXY_LIT("/") + dsl::p<attr_types> + LEXY_LIT(":") + dsl::p<name>;

                return dsl::peek(attr_w_name) >> attr_w_name
                     | dsl::peek(type_w_name) >> dsl::p<name> + LEXY_LIT(":") + dsl::nullopt + dsl::p<name>
                     | dsl::else_ >> dsl::nullopt + dsl::nullopt + just_name;
            }();

            static constexpr auto value = lexy::construct<col_name_info>;
        };
    }
}

#endif //FLEXFRINGE_CSVHEADER_H
