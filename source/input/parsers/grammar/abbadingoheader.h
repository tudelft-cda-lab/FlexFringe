//
// Created by tom on 1/13/23.
//

#ifndef FLEXFRINGE_ABBADINGOHEADER_H
#define FLEXFRINGE_ABBADINGOHEADER_H

#include <lexy/dsl/literal.hpp>
#include <lexy/action/parse.hpp> // lexy::parse
#include <lexy/callback.hpp>     // value callbacks
#include <lexy/dsl.hpp>          // lexy::dsl::*

#include <string>
#include <utility>
#include <vector>
#include <set>
#include <iostream>

struct abbadingo_attribute {
    std::string name;
    std::set<char> type;

    abbadingo_attribute(const std::string& type, std::string name)
            : name(std::move(name))
            , type(std::begin(type), std::end(type))
            {}

    friend std::ostream& operator << (std::ostream &os, const abbadingo_attribute &self) {
        for (auto x: self.type) {
            os << x;
        }
        os <<  "/" << self.name;
        return os;
    }
};

struct abbadingo_header_part {
    uint64_t number;
    std::vector<abbadingo_attribute> attributes;

    abbadingo_header_part (uint64_t number, std::vector<abbadingo_attribute> attrs)
            : number(number)
            , attributes(std::move(attrs)) {}

    explicit abbadingo_header_part (uint64_t number)
            : number(number)
            , attributes(std::vector<abbadingo_attribute>{}) {}

    friend std::ostream& operator << (std::ostream &os, const abbadingo_header_part &self) {
        os << self.number;
        if (!self.attributes.empty()) {
            os << ":";
        }
        unsigned long count = 0;
        for (auto& attr: self.attributes) {
            os << attr;
            count += 1;
            if (count < self.attributes.size()) {
                os << ",";
            }
        }
        return os;
    }
};

struct abbadingo_header_info {
    abbadingo_header_part traces;
    abbadingo_header_part symbols;

    friend std::ostream& operator << (std::ostream &os, const abbadingo_header_info &self) {
        os << self.traces << " " << self.symbols;
        return os;
    }
};

namespace {
    struct test {
        std::string a;
        std::string b;
        std::string c;
    };

    namespace grammar {
        namespace dsl = lexy::dsl;

        struct number {
            static constexpr auto rule = dsl::integer<std::uint64_t>;
            static constexpr auto value = lexy::as_integer<std::uint64_t>;
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

            static constexpr auto value = lexy::as_string<std::string>;
        };

        // Attribute name
        struct attr_name {
            static constexpr auto rule = dsl::identifier(dsl::ascii::word);
            static constexpr auto value = lexy::as_string<std::string>;
        };

        // Attribute definition: types/attribute_name
        struct attr_def {
            static constexpr auto rule = dsl::p<attr_types> + LEXY_LIT("/") + dsl::p<attr_name>;
            static constexpr auto value = lexy::construct<abbadingo_attribute>;
        };

        // List of attribute definitions: :dsft/name1,dsft/name2,...
        struct attr_list {
            static constexpr auto rule = LEXY_LIT(":") + dsl::list(dsl::p<attr_def>, dsl::sep(dsl::comma));
            static constexpr auto value = lexy::as_list<std::vector<abbadingo_attribute>>;
        };

        // One out of two abbadingo header parts: number:dsft/name1,dsft/name2,...
        struct abbadingo_header_part_p {
            static constexpr auto rule = [] {
                // Do we have a : before the next whitespace?
                auto attr_lookahead = dsl::lookahead(LEXY_LIT(":"), LEXY_LIT(" "));
                return (attr_lookahead >> (dsl::p<number> + dsl::p<attr_list>) | dsl::else_ >> dsl::p<number>);
            }();

            static constexpr auto value = lexy::construct<abbadingo_header_part>;
        };

        // The complete abbadingo header: number:dsft/name1,... number:dsft/name2,...
        struct abbadingo_header_p {
            static constexpr auto rule = dsl::twice(dsl::p<abbadingo_header_part_p>, dsl::trailing_sep(dsl::ascii::space))
                    + dsl::whitespace(dsl::ascii::blank)
                    + (dsl::newline | dsl::eof);
            static constexpr auto value = lexy::construct<abbadingo_header_info>;
        };
    }
}



#endif //FLEXFRINGE_ABBADINGOHEADER_H
