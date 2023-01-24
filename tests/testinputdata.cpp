#include "catch.hpp"

#include "input/inputdata.h"
#include "input/inputdatalocator.h"
#include "input/tail.h"
#include "input/parsers/abbadingoparser.h"
#include "input/parsers/csvparser.h"

#include "csv.hpp"

#include <cstdio>
#include <iostream>
#include <sstream>

using Catch::Matchers::Equals;

TEST_CASE("AbbadingoReader: smoke test", "[parsing]") {

    std::string input = "2 50\n"
                        "1 3 12 26 29\n"
                        "0 11 36 9 3 11 17 20 34 20 20 20 10\n";
    std::istringstream input_stream(input);

    auto input_data = inputdata();
    inputdata_locator::provide(&input_data);

    auto parser = abbadingoparser(input_stream);
    input_data.read(&parser);

    std::list<std::string> expected_traces = {
            "1 3 12 26 29",
            "0 11 36 9 3 11 17 20 34 20 20 20 10"
    };

    for (auto trace: input_data) {
        auto expected = expected_traces.front();
        auto actual = trace->to_string();
        REQUIRE_THAT(actual, Equals(expected));
        expected_traces.pop_front();
    }
}


TEST_CASE("CSVReader: smoke test", "[parsing]") {
    // This tests makes sure that having spaces in front of the column names in the header
    // does not break things, as it caused issues recognizing the special column types before
    // (e.g. [timestamp, id, symb] vs [timestamp,id,symb])

    std::string input_whitespace = "id, symb\n"
                                   "670edd27, Received symbol b\n"
                                   "670edd27, Received symbol b\n"
                                   "670edd28, Received symbol b";
    std::istringstream input(input_whitespace);

    auto input_data = inputdata();
    inputdata_locator::provide(&input_data);

    auto parser = csv_parser(input,
                             csv::CSVFormat().trim({' '}));

    input_data.read(&parser);

    std::list<std::string> expected_traces = {
            "0 2 Received symbol b Received symbol b",
            "0 1 Received symbol b"
    };

    for (auto trace: input_data) {
        auto expected = expected_traces.front();
        auto actual = trace->to_string();
        REQUIRE_THAT(actual, Equals(expected));
        expected_traces.pop_front();
    }

}

TEST_CASE("CSVReader: Special characters", "[parsing]") {
    // check if we can parse csv files with the abbadingo delimiter symbols

    std::string input_whitespace = "id, symb\n"
                                   "1, msg: b\n"
                                   "1, msg: b\n"
                                   "2, msg: a/b";
    std::istringstream input(input_whitespace);

    auto input_data = inputdata();
    inputdata_locator::provide(&input_data);

    auto parser = csv_parser(input,
                             csv::CSVFormat().trim({' '}));

    input_data.read(&parser);

    std::list<std::string> expected_traces = {
            "0 2 msg: b msg: b",
            "0 1 msg: a/b",
    };

    for (auto trace: input_data) {
        auto expected = expected_traces.front();
        auto actual = trace->to_string();
        REQUIRE_THAT(actual, Equals(expected));
        expected_traces.pop_front();
    }
}

TEST_CASE("CSVReader: Sliding window csv", "[parsing]") {
    // check if we can parse csv files with the abbadingo delimiter symbols

    std::string input_whitespace = "id, symb\n"
                                   "1, a\n"
                                   "1, b\n"
                                   "1, c\n"
                                   "1, d";
    std::istringstream input(input_whitespace);

    auto input_data = inputdata();
    inputdata_locator::provide(&input_data);

    auto parser = csv_parser(input,
                             csv::CSVFormat().trim({' '}));

    input_data.read_slidingwindow(&parser,
                                  4,
                                  1,
                                  false);

    std::list<std::string> expected_traces = {
            "0 3 a b c",
            "0 3 b c d"
    };

    for (auto trace: input_data) {
        auto expected = expected_traces.back();
        auto actual = trace->to_string();
        REQUIRE_THAT(actual, Equals(expected));
        expected_traces.pop_back();
    }
}

TEST_CASE("CSVReader: Sliding window abbadingo", "[parsing]") {
    // check if we can parse csv files with the abbadingo delimiter symbols

    std::string input_whitespace = "1 4\n"
                                   "0 4 a b c d";
    std::istringstream input(input_whitespace);

    auto input_data = inputdata();
    inputdata_locator::provide(&input_data);

    auto parser = abbadingoparser(input);
    input_data.read_slidingwindow(&parser,
                                  4,
                                  1,
                                  false);

    std::list<std::string> expected_traces = {
            "0 3 a b c",
            "0 3 b c d"
    };

    for (auto trace: input_data) {
        auto expected = expected_traces.back();
        auto actual = trace->to_string();
        REQUIRE_THAT(actual, Equals(expected));
        expected_traces.pop_back();
    }
}


TEST_CASE("CSVReader: symbol attributes", "[parsing]") {
    // check if we can parse csv files with the abbadingo delimiter symbols

    std::string input_whitespace = "id, symb, attr/d:test\n"
                                   "1, a, 1.0\n"
                                   "1, b, 2.0\n"
                                   "2, c, 3.0";
    std::istringstream input(input_whitespace);

    auto input_data = inputdata();
    inputdata_locator::provide(&input_data);

    auto parser = csv_parser(input,
                             csv::CSVFormat().trim({' '}));

    input_data.read(&parser);

    std::vector<trace*> actual_traces(input_data.begin(), input_data.end());

    REQUIRE(actual_traces.size() == 2);

    //TODO: make sure this value actually gets set :/
    REQUIRE(actual_traces.at(0)->get_head()->td->attr[0] == 1.0);
}