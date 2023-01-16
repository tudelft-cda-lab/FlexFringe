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


namespace {
    namespace symbol_grammar {
        namespace dsl = lexy::dsl;

        struct number {
            static constexpr auto rule = dsl::integer<std::uint64_t>;
            static constexpr auto value = lexy::as_integer<std::uint64_t>;
        };

        // Attribute value
        struct attr_val {
            static constexpr auto rule = dsl::capture(dsl::digits<>) + dsl::period + dsl::capture(dsl::digits<>);
            static constexpr auto value = lexy::callback<double>([](auto integer, auto decimal) {
                std::string tmp(integer.begin(), integer.end());
                tmp.append(".");
                tmp.append(decimal.begin(), decimal.end());
                return std::stod(tmp);
            });
        };

        struct symbol {
            static constexpr auto rule = dsl::p<number>;
        };
    }
}



#endif //FLEXFRINGE_ABBADINGOSYMBOL_H
