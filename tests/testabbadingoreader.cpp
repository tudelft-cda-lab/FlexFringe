
#include "catch.hpp"
#include "input/abbadingoreader.h"
#include "input/inputdatalocator.h"
#include <cstdio>
#include <sstream>
#include <iostream>

using Catch::Matchers::Equals;

TEST_CASE("AbbadingoReader: smoke test", "[parsing]") {

    std::string input = "2 50\n"
                        "1 2 12 26\n"
                        "0 14 36 9 3 11 17 20 34 20 20 20 10\n";
    std::istringstream input_stream(input);

    auto reader = AbbadingoInputData();
    InputDataLocator::provide(&reader);

    reader.read(input_stream);

    for (auto trace: reader) {
        std::cout << trace->to_string() << std::endl;
    }

}
