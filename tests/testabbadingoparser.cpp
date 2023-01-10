
#include "catch.hpp"
#include "input/parsers/abbadingoparser.h"
#include <cstdio>
#include <iostream>
#include <sstream>

TEST_CASE("abbadingo_parser: smoke test", "[parsing]") {
    std::string input = "2 2\n"
                        "0 2 a b\n"
                        "1 2 c d";
    std::istringstream inputstream(input);

    auto parser = abbadingoparser(inputstream);

    auto first = parser.next().value();
    REQUIRE(first.get("id") == std::vector<std::string> {"0"});
    REQUIRE(first.get("symb") == std::vector<std::string> {"a"});
    REQUIRE(first.get("type") == std::vector<std::string> {"0"});

    auto second = parser.next().value();
    REQUIRE(second.get("id") == std::vector<std::string> {"0"});
    REQUIRE(second.get("symb") == std::vector<std::string> {"b"});
    REQUIRE(second.get("type") == std::vector<std::string> {"0"});

    auto third = parser.next().value();
    REQUIRE(third.get("id") == std::vector<std::string> {"1"});
    REQUIRE(third.get("symb") == std::vector<std::string> {"c"});
    REQUIRE(third.get("type") == std::vector<std::string> {"1"});

    auto fourth = parser.next().value();
    REQUIRE(fourth.get("id") == std::vector<std::string> {"1"});
    REQUIRE(fourth.get("symb") == std::vector<std::string> {"d"});
    REQUIRE(fourth.get("type") == std::vector<std::string> {"1"});
}

