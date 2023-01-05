
#include "catch.hpp"
#include "input/parsers/csvparser.h"
#include <cstdio>
#include <iostream>

using Catch::Matchers::Equals;

TEST_CASE( "CSVHeaderParser: id", "[parsing]" ) {
    std::vector<std::string> input = {"id:column_a", "column_b", "column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("id") == std::set<int> {0});
}

TEST_CASE( "CSVHeaderParser: multiple id", "[parsing]" ) {
    std::vector<std::string> input = {"id:column_a", "id:column_b", "column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("id") == std::set<int> {0, 1});
}

TEST_CASE( "CSVHeaderParser: multiple id with skip", "[parsing]" ) {
    std::vector<std::string> input = {"id:column_a", "column_b", "id:column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("id") == std::set<int> {0, 2});
}

TEST_CASE( "CSVHeaderParser: custom label", "[parsing]" ) {
    std::vector<std::string> input = {"special:column_a", "column_b", "special:column_c"};

    auto parser = csv_header_parser(input, {"special"});

    REQUIRE(parser.get("special") == std::set<int> {0, 2});
}

TEST_CASE( "CSVHeaderParser: implicit label", "[parsing]" ) {
    std::vector<std::string> input = {"id:column_a", "column_b", "special:column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get("id") == std::set<int> {0});
    REQUIRE(parser.get("special") == std::set<int> {2});
}

TEST_CASE( "CSVHeaderParser: get symbol attributes", "[parsing]" ) {
    std::vector<std::string> input = {"attr:column_a", "column_b", "attr:column_c"};

    auto parser = csv_header_parser(input);

    REQUIRE(parser.get_names("attr") == std::vector<std::string> {"column_a", "column_c"});
}