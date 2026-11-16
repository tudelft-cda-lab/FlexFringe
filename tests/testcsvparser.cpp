
#include "catch.hpp"
#include "input/parsers/csvparser.h"
#include <cstdio>
#include <sstream>
#include <iostream>
#include <sstream>

TEST_CASE("csv_parser: smoke test", "[parsing]") {
    std::string input = "ff_id:id, ff_symb:symbol\n"
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
    REQUIRE(first.get("ff_id") == std::vector<std::string>{"1"});
    REQUIRE(first.get("ff_symb") == std::vector<std::string>{"a"});

    auto second = parser.next().value();
    REQUIRE(second.get("ff_id") == std::vector<std::string>{"2"});
    REQUIRE(second.get("ff_symb") == std::vector<std::string>{"b"});

    auto third = parser.next().value();
    REQUIRE(third.get("ff_id") == std::vector<std::string>{"3"});
    REQUIRE(third.get("ff_symb") == std::vector<std::string>{"c"});
}


TEST_CASE("csv_parser: smoke test - only labels", "[parsing]") {
    std::string input_str = "ff_id, ff_symb\n"
                            "670edd27, Received symbol b\n"
                            "670edd27, Received symbol b\n"
                            "670edd28, Received symbol b";
    std::istringstream input(input_str);

    auto parser = csv_parser(input,
                             csv::CSVFormat().trim({' '}));

    auto first = parser.next().value();
    REQUIRE(first.get("ff_id") == std::vector<std::string>{"670edd27"});
    REQUIRE(first.get("ff_symb") == std::vector<std::string>{"Received symbol b"});

    auto second = parser.next().value();
    REQUIRE(second.get("ff_id") == std::vector<std::string>{"670edd27"});
    REQUIRE(second.get("ff_symb") == std::vector<std::string>{"Received symbol b"});

    auto third = parser.next().value();
    REQUIRE(third.get("ff_id") == std::vector<std::string>{"670edd28"});
    REQUIRE(third.get("ff_symb") == std::vector<std::string>{"Received symbol b"});
}

TEST_CASE("csv_parser: smoke test with attr", "[parsing]") {
    std::string input = "ff_id, ff_symb, ff_sattr:d/test\n"
                        "1, a, 1.0\n"
                        "2, b, 2.0\n"
                        "3, c, 3.0";
    std::istringstream inputstream(input);

    auto parser = csv_parser(
            std::make_unique<csv::CSVReader>(
                    inputstream,
                    csv::CSVFormat().trim({' '})
            )
    );

    auto first = parser.next().value();
    REQUIRE(first.get_str("ff_id") == "1");
    REQUIRE(first.get_str("ff_symb") == "a");
    REQUIRE(first.get_symb_attr_info().at(0).get_value() == "1.0");
    REQUIRE(first.get_symb_attr_info().at(0).get_name() == "test");
    REQUIRE(first.get_symb_attr_info().at(0).is_discrete());

    auto second = parser.next().value();
    REQUIRE(second.get_str("ff_id") == "2");
    REQUIRE(second.get_str("ff_symb") == "b");
    REQUIRE(second.get_symb_attr_info().at(0).get_value() == "2.0");
    REQUIRE(second.get_symb_attr_info().at(0).get_name() == "test");
    REQUIRE(second.get_symb_attr_info().at(0).is_discrete());

    auto third = parser.next().value();
    REQUIRE(third.get_str("ff_id") == "3");
    REQUIRE(third.get_str("ff_symb") == "c");
    REQUIRE(third.get_symb_attr_info().at(0).get_value() == "3.0");
    REQUIRE(third.get_symb_attr_info().at(0).get_name() == "test");
    REQUIRE(third.get_symb_attr_info().at(0).is_discrete());
}

TEST_CASE("csv_parser: smoke test with tattr", "[parsing]") {
    std::string input = "ff_id, ff_symb, ff_tattr:d/test\n"
                        "1, a, 1.0\n"
                        "1, b,    \n"
                        "3, c, 3.0";
    std::istringstream inputstream(input);

    auto parser = csv_parser(
            std::make_unique<csv::CSVReader>(
                    inputstream,
                    csv::CSVFormat().trim({' '})
            )
    );

    auto first = parser.next().value();
    auto first_tattr = first.get_trace_attr_info();
    REQUIRE(first.get_str("ff_id") == "1");
    REQUIRE(first.get_str("ff_symb") == "a");
    REQUIRE(first_tattr->at(0).get_value() == "1.0");

    auto second = parser.next().value();
    auto second_tattr = second.get_trace_attr_info();
    REQUIRE(second.get_str("ff_id") == "1");
    REQUIRE(second.get_str("ff_symb") == "b");
    REQUIRE(second_tattr->at(0).get_value() == "1.0");

    REQUIRE(second_tattr == first_tattr);
    REQUIRE(second_tattr->size() == 1);

    auto third = parser.next().value();
    auto third_tattr = third.get_trace_attr_info();
    REQUIRE(third.get_str("ff_id") == "3");
    REQUIRE(third.get_str("ff_symb") == "c");
    REQUIRE(third_tattr->at(0).get_value() == "3.0");
}

TEST_CASE("csv_parser: smoke test with duplicate tattr", "[parsing]") {
    std::string input = "ff_id, ff_symb, ff_tattr:d/test\n"
                        "1, a, 1.0\n"
                        "1, b, 2.0\n" // <- This is not allowed, this redefines the trace attribute "test" for trace with id 1
                        "3, c, 3.0";
    std::istringstream inputstream(input);

    auto parser = csv_parser(
            std::make_unique<csv::CSVReader>(
                    inputstream,
                    csv::CSVFormat().trim({' '})
            )
    );

    auto first = parser.next().value();
    auto first_tattr = first.get_trace_attr_info();
    REQUIRE(first.get_str("ff_id") == "1");
    REQUIRE(first.get_str("ff_symb") == "a");
    REQUIRE(first_tattr->at(0).get_value() == "1.0");

    REQUIRE_THROWS_WITH(parser.next().value(), "Error: duplicate trace attribute value \"test\" specified for trace with id: 1");
}



