
#include "catch.hpp"
#include "input/parsers/csvparser.h"
#include <cstdio>
#include <sstream>
#include <iostream>

TEST_CASE("csv_parser: smoke test", "[parsing]") {
    std::string input = "id:id, symb:symbol\n"
                        "1, a\n"
                        "2, b\n"
                        "3, c";
    std::istringstream inputstream(input);

    auto parser = csv_parser(
            std::make_unique<csv::CSVReader>(inputstream)
    );

    auto first = parser.next().value();
    REQUIRE(first.get("id") == std::vector<std::string> {"1"});

    auto second = parser.next().value();
    REQUIRE(second.get("id") == std::vector<std::string> {"2"});

    auto third = parser.next().value();
    REQUIRE(third.get("id") == std::vector<std::string> {"3"});
}
