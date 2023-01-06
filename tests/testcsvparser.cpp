
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
            std::make_unique<csv::CSVReader>(
                    inputstream,
                    csv::CSVFormat().trim({' '})
                    )
    );

    auto first = parser.next().value();
    REQUIRE(first.get("id") == std::vector<std::string> {"1"});
    REQUIRE(first.get("symb") == std::vector<std::string> {"a"});

    auto second = parser.next().value();
    REQUIRE(second.get("id") == std::vector<std::string> {"2"});
    REQUIRE(second.get("symb") == std::vector<std::string> {"b"});

    auto third = parser.next().value();
    REQUIRE(third.get("id") == std::vector<std::string> {"3"});
    REQUIRE(third.get("symb") == std::vector<std::string> {"c"});
}
