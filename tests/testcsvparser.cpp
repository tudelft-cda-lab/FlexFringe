
#include "catch.hpp"
#include "input/parsers/csvparser.h"
#include <cstdio>
#include <sstream>
#include <iostream>

using Catch::Matchers::Equals;

TEST_CASE("csv_parser: smoke test", "[parsing]") {
    std::string input = "id, symb\n"
                        "1, b\n"
                        "1, b\n"
                        "1, b";
    std::istringstream inputstream(input);

    auto parser = csv_parser(
            std::make_unique<csv::CSVReader>(inputstream)
    );

    parser.parse();
}
