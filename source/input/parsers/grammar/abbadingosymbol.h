//
// Created by tom on 1/13/23.
//

#ifndef FLEXFRINGE_ABBADINGOSYMBOL_H
#define FLEXFRINGE_ABBADINGOSYMBOL_H

#include <lexy/dsl/literal.hpp>
#include <lexy/action/parse.hpp> // lexy::parse
#include <lexy/callback.hpp>     // value callbacks
#include <lexy/dsl.hpp>          // lexy::dsl::*

#include <string>
#include <utility>
#include <vector>
#include <set>
#include <iostream>


struct abbadingo_symbol_info {
    uint64_t number;
    std::optional<std::vector<std::string>> attribute_values;
    std::optional<std::string> data;
};

namespace {
    namespace symbol_grammar {
        namespace dsl = lexy::dsl;

        struct number {
            static constexpr auto rule = dsl::integer<std::uint64_t>;
            static constexpr auto value = lexy::as_integer<std::uint64_t>;
        };

        // Attribute value - convert to double
        struct attr_val_double {
            static constexpr auto rule = dsl::capture(dsl::digits<>) + dsl::period + dsl::capture(dsl::digits<>);
            static constexpr auto value = lexy::callback<double>([](auto integer, auto decimal) {
                std::string tmp(integer.begin(), integer.end());
                tmp.append(".");
                tmp.append(decimal.begin(), decimal.end());
                return std::stod(tmp);
            });
        };

        // Attribute value - keep as string
        struct attr_val_str {
            static constexpr auto rule = dsl::capture(dsl::digits<>) + dsl::period + dsl::capture(dsl::digits<>);
            static constexpr auto value = lexy::callback<std::string>([](auto integer, auto decimal) {
                std::string tmp(integer.begin(), integer.end());
                tmp.append(".");
                tmp.append(decimal.begin(), decimal.end());
                return tmp;
            });
        };

        // Attribute value list
        struct attr_val_list {
            static constexpr auto rule = LEXY_LIT(":") + dsl::list(dsl::p<attr_val_str>, dsl::sep(dsl::comma));
            static constexpr auto value = lexy::as_list<std::vector<std::string>>;
        };

        // Data
        struct data {
            static constexpr auto rule = LEXY_LIT("/") + dsl::identifier(dsl::ascii::alpha_digit_underscore);
            static constexpr auto value = lexy::as_string<std::string>;
        };

        struct symbol {
            static constexpr auto rule = [] {
                auto lookahead_attr = dsl::lookahead(LEXY_LIT(":"), dsl::literal_set(LEXY_LIT(" "), LEXY_LIT("\n")));
                auto lookahead_data = dsl::lookahead(LEXY_LIT("/"), dsl::literal_set(LEXY_LIT(" "), LEXY_LIT("\n")));
                return  (dsl::p<number>
                        + (lookahead_attr >> dsl::p<attr_val_list> | dsl::else_ >> dsl::nullopt)
                        + (lookahead_data >> dsl::p<data> | dsl::else_ >> dsl::nullopt));
            }();

            static constexpr auto value = lexy::callback<abbadingo_symbol_info>([](auto number, auto attr_val_list, auto data) {
                return abbadingo_symbol_info {
                    .number = number,
                    .attribute_values = attr_val_list,
                    .data = data
                };
            });
        };

        struct symbol_list {
            static constexpr auto rule = dsl::list(dsl::p<symbol>, dsl::sep(dsl::ascii::space));
            static constexpr auto value = lexy::as_list<std::vector<abbadingo_symbol_info>>;
        };

        struct abbadingo_trace {
            static constexpr auto rule = dsl::p<number> + dsl::p<symbol> + dsl::p<symbol_list>;
            static constexpr auto value = lexy::callback<std::string>([] (auto a, auto b, auto c) {
                return "TODO";
            });
        };
    }
}



#endif //FLEXFRINGE_ABBADINGOSYMBOL_H
