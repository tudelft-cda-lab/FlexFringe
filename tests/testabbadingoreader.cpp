
#include "catch.hpp"
#include "input/abbadingoreader.h"
#include "input/inputdatalocator.h"
#include <cstdio>
#include <sstream>
#include <iostream>

using Catch::Matchers::Equals;

TEST_CASE("AbbadingoReader: smoke test", "[parsing]") {

    std::string input = "2 50\n"
                        "1 3 12 26 29\n"
                        "0 11 36 9 3 11 17 20 34 20 20 20 10\n";
    std::istringstream input_stream(input);

    auto reader = abbadingo_inputdata();
    inputdata_locator::provide(&reader);

    reader.read(input_stream);

    std::list<std::string> expected_traces = {
            "1 3 12 26 29",
            "0 11 36 9 3 11 17 20 34 20 20 20 10"
    };

    for (auto trace: reader) {
        auto expected = expected_traces.front();
        auto actual = trace->to_string();
        REQUIRE_THAT(actual, Equals(expected));
        expected_traces.pop_front();
    }
}
