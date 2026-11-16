
#include "catch.hpp"
#include "input/parsers/csvparser.h"
#include "lexy/input/string_input.hpp"
#include "lexy/action/parse.hpp"
#include "lexy/action/parse_as_tree.hpp"
#include "lexy_ext/report_error.hpp"
#include "input/parsers/grammar/csvheader.h"
#include "lexy/action/trace.hpp"
#include <cstdio>
#include <iostream>
#include <sstream>

using Catch::Matchers::Equals;

TEST_CASE( "CSVHeaderParser: id", "[parsing]" ) {
    std::vector<std::string> input = {"ff_id:column_a", "column_b", "column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("ff_id") == std::set<int> {0});
}

TEST_CASE( "CSVHeaderParser: multiple id", "[parsing]" ) {
    std::vector<std::string> input = {"ff_id:column_a", "ff_id:column_b", "column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("ff_id") == std::set<int> {0, 1});
}

TEST_CASE( "CSVHeaderParser: multiple id with skip", "[parsing]" ) {
    std::vector<std::string> input = {"ff_id:column_a", "column_b", "ff_id:column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("ff_id") == std::set<int> {0, 2});
}

TEST_CASE( "CSVHeaderParser: multiple id with symbol", "[parsing]" ) {
    std::vector<std::string> input = {"ff_id:column_a", "ff_symb:column_b", "ff_id:column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("ff_id") == std::set<int> {0, 2});
    REQUIRE(parser.get("ff_symb") == std::set<int> {1});
}

TEST_CASE( "CSVHeaderParser: custom label", "[parsing]" ) {
    std::vector<std::string> input = {"special:column_a", "column_b", "special:column_c"};

    auto parser = csv_header_parser(input, {"special"});

    REQUIRE(parser.get("special") == std::set<int> {0, 2});
}

// Not sure if we should allow implicit labels or not
TEST_CASE( "CSVHeaderParser: implicit label", "[parsing]" ) {
    std::vector<std::string> input = {"ff_id:column_a", "column_b", "special:column_c"};

    CHECK_THROWS(csv_header_parser(input));
}

TEST_CASE( "CSVHeaderParser: get symbol attributes", "[parsing]" ) {
    std::vector<std::string> input = {"ff_sattr:d/column_a", "column_b", "ff_sattr:f/column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get_names("ff_sattr") == std::vector<std::string> {"column_a", "column_c"});
}

TEST_CASE( "CSVHeaderParser: id and symbol", "[parsing]" ) {
    std::vector<std::string> input = {"ff_id:id", "ff_symb:symbol"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("ff_id") == std::set<int> {0});
    REQUIRE(parser.get("ff_symb") == std::set<int> {1});
}

TEST_CASE( "CSVHeaderParser: no :, just label", "[parsing]" ) {
    std::vector<std::string> input = {"ff_id", "ff_symb"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("ff_id") == std::set<int> {0});
    REQUIRE(parser.get("ff_symb") == std::set<int> {1});
}

TEST_CASE("CSVHeaderParser: column name, just name", "[parsing]") {
    auto input = lexy::zstring_input("example_name");
    auto result = lexy::parse<csv_header_grammar::col_name>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.name == "example_name");
    REQUIRE(!value.attr_types.has_value());
    REQUIRE(!value.type_name.has_value());
}

TEST_CASE("CSVHeaderParser: column name, with column type specifier", "[parsing]") {
    auto input = lexy::zstring_input("ff_symb:example_name");
    auto result = lexy::parse<csv_header_grammar::col_name>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.name == "example_name");
    REQUIRE(!value.attr_types.has_value());
    REQUIRE(value.type_name.has_value());
    REQUIRE(value.type_name.value() == "ff_symb");
}

TEST_CASE("CSVHeaderParser: column name, with attribute specifier", "[parsing]") {
    auto input = lexy::zstring_input("ff_sattr:dsft/example_name");
    auto result = lexy::parse<csv_header_grammar::col_name>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.name == "example_name");
    REQUIRE(value.attr_types.has_value());
    REQUIRE(value.attr_types.value() == std::set<std::string> {"d", "s", "f", "t"});
    REQUIRE(value.type_name.has_value());
    REQUIRE(value.type_name.value() == "ff_sattr");
}

TEST_CASE("CSVHeaderParser: column name, incomplete attr spec", "[parsing]") {
    auto input = lexy::zstring_input("ff_sattr:example_name");
    auto result = lexy::parse<csv_header_grammar::col_name>( input, lexy_ext::report_error);
    REQUIRE(!result.has_value());
}

TEST_CASE("CSVHeaderParser: column name, incomplete attr spec 2", "[parsing]") {
    auto input = lexy::zstring_input("ff_sattr:dsft");
    auto result = lexy::parse<csv_header_grammar::col_name>(input, lexy_ext::report_error);
    REQUIRE(!result.has_value());
}

TEST_CASE("CSVHeaderParser: column name, duplicate symbol attribute name", "[parsing]") {
    std::vector<std::string> input = {"ff_sattr:dsft/example_name", "ff_sattr:dsft/example_name"};
    REQUIRE_THROWS_WITH(csv_header_parser(input), "Duplicate attribute name: example_name");
}

TEST_CASE("CSVHeaderParser: column name, duplicate trace attribute name", "[parsing]") {
    std::vector<std::string> input = {"ff_tattr:dsft/example_name", "ff_tattr:dsft/example_name"};
    REQUIRE_THROWS_WITH(csv_header_parser(input), "Duplicate attribute name: example_name");
}

TEST_CASE("CSVHeaderParser: column name, duplicate other name", "[parsing]") {
    std::vector<std::string> input = {"ff_id", "ff_id"};
    auto result = csv_header_parser(input);
    REQUIRE(result.get("ff_id") == std::set<int>{0, 1});
}