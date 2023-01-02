
#include "catch.hpp"
#include "input/csvreader.h"
#include <cstdio>
#include <sstream>
#include <iostream>

using Catch::Matchers::Equals;

TEST_CASE( "CSVReader: smoke test", "[parsing]" ) {
    // This tests makes sure that having spaces in front of the column names in the header
    // does not break things, as it caused issues recognizing the special column types before
    // (e.g. [timestamp, id, symb] vs [timestamp,id,symb])

    std::string input_whitespace = "id, symb\n"
                                   "670edd27, Received symbol b\n"
                                   "670edd27, Received symbol b\n"
                                   "670edd28, Received symbol b";
    std::istringstream input_stream_whitespace(input_whitespace);

    auto reader = csv_inputdata();
    inputdata_locator::provide(&reader);

    reader.read(input_stream_whitespace);

    std::list<std::string> expected_traces = {
            "0 1 Received symbol b",
            "0 2 Received symbol b Received symbol b"
    };

    for (auto trace: reader) {
        auto expected = expected_traces.front();
        auto actual = trace->to_string();
        REQUIRE_THAT(actual, Equals(expected));
        expected_traces.pop_front();
    }

}

TEST_CASE( "CSVReader: Special characters", "[parsing]" ) {
    // check if we can parse csv files with the abbadingo delimiter symbols

    std::string input_whitespace = "id, symb\n"
                                   "1, msg: b\n"
                                   "1, msg: b\n"
                                   "2, msg: a/b";
    std::istringstream input_stream_whitespace(input_whitespace);

    auto reader = csv_inputdata();
    inputdata_locator::provide(&reader);

    reader.read(input_stream_whitespace);

    std::list<std::string> expected_traces = {
            "0 1 msg: a/b",
            "0 2 msg: b msg: b"
    };

    for (auto trace: reader) {
        auto expected = expected_traces.front();
        auto actual = trace->to_string();
        REQUIRE_THAT(actual, Equals(expected));
        expected_traces.pop_front();
    }

}