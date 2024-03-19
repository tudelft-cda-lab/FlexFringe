
#include "catch.hpp"
#include "input/parsers/abbadingoparser.h"
#include <cstdio>
#include <iostream>
#include <sstream>
#include <lexy/action/parse.hpp>
#include <lexy/action/parse_as_tree.hpp>
#include <lexy/action/trace.hpp>
#include <lexy/visualize.hpp>
#include <lexy_ext/report_error.hpp>
#include <lexy/input/string_input.hpp>
#include <lexy/action/match.hpp>
#include "input/parsers/grammar/abbadingoheader.h"
#include "input/parsers/grammar/abbadingosymbol.h"
#include "input/inputdatalocator.h"

TEST_CASE("abbadingo header parser: number", "[parsing]") {
    auto input = lexy::zstring_input("123");
    auto result = lexy::parse<grammar::number>(input, lexy_ext::report_error);
    CHECK(result.has_value());
}

TEST_CASE("abbadingo header parser: attribute types", "[parsing]") {
    auto input = lexy::zstring_input("dsft");
    auto result = lexy::parse<grammar::attr_types>(input, lexy_ext::report_error);
    CHECK(result.has_value());
    std::cout << "value: " << result.value() << "\n";
}

TEST_CASE("abbadingo header parser: illegal attribute types", "[parsing]") {
    auto input = lexy::zstring_input("abcd");
    REQUIRE_FALSE(
        lexy::match<grammar::attr_types>(input, lexy_ext::report_error)
    );
}

TEST_CASE("abbadingo header parser: single attribute type/name", "[parsing]") {
    auto input = lexy::zstring_input("dsft/thisisthename");
    auto result = lexy::parse<grammar::attr_def>(input, lexy_ext::report_error);

    REQUIRE(result.has_value());
    auto value = result.value();

    REQUIRE(value.name == "thisisthename");
    REQUIRE(value.type == std::set<char> {'d', 's', 'f', 't'});
}

TEST_CASE("abbadingo header parser: attribute list", "[parsing]") {
    auto input = lexy::zstring_input(":ds/name1,ft/name2");
    auto result = lexy::parse<grammar::attr_list>(input, lexy_ext::report_error);

    REQUIRE(result.has_value());
    std::vector<abbadingo_attribute> value = result.value();

    REQUIRE(value.front().type == std::set<char> {'d', 's'});
    REQUIRE(value.front().name == "name1");

    REQUIRE(value.back().type == std::set<char> {'f', 't'});
    REQUIRE(value.back().name == "name2");
}

TEST_CASE("abbadingo header parser: header part", "[parsing]") {
    auto input = lexy::zstring_input("50:dsft/name1,dsft/name2");
    auto result = lexy::parse<grammar::abbadingo_header_part_p>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    //TODO check value
    std::cout << "value: " << value << "\n";
}

TEST_CASE("abbadingo header parser: header part, no attributes", "[parsing]") {
    auto input = lexy::zstring_input("50");
    auto result = lexy::parse<grammar::abbadingo_header_part_p>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    //TODO check value
    std::cout << "value: " << value << "\n";
}

TEST_CASE("abbadingo header parser: whole thing", "[parsing]") {
    auto input = lexy::zstring_input("50:ds/name1,ft/name2 8:d/n1,f/n2");
    auto result = lexy::parse<grammar::abbadingo_header_p>(input, lexy_ext::report_error);
    REQUIRE(result);
    auto value = result.value();
    //TODO check value
    std::cout << "value: " << value << "\n";
}

TEST_CASE("abbadingo header parser: trailing whitespace", "[parsing]") {
    auto input = lexy::zstring_input("50:ds/name1,ft/name2 8:d/n1,f/n2           ");
    auto result = lexy::parse<grammar::abbadingo_header_p>(input, lexy_ext::report_error);
    REQUIRE(result);
    auto value = result.value();
}

TEST_CASE("abbadingo symbol parser", "[parsing]") {
    auto input = lexy::zstring_input("10.1");
    auto result = lexy::parse<symbol_grammar::attr_val_double>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value == 10.1);
}

TEST_CASE("abbadingo symbol parser 1", "[parsing]") {
    auto input = lexy::zstring_input("10:1.23/asdf");
    auto result = lexy::parse<symbol_grammar::symbol>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.name == "10");
    REQUIRE(value.attribute_values.value().front() == "1.23");
    REQUIRE(value.data.value() == "asdf");
}

TEST_CASE("abbadingo symbol parser: only attr", "[parsing]") {
    auto input = lexy::zstring_input("10:1.23");
    auto result = lexy::parse<symbol_grammar::symbol>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.name == "10");
    REQUIRE(value.attribute_values.value().front() == "1.23");
    REQUIRE(!value.data.has_value());
}

TEST_CASE("abbadingo symbol parser: only data", "[parsing]") {
    auto input = lexy::zstring_input("10/blabla");
    auto result = lexy::parse<symbol_grammar::symbol>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.name == "10");
    REQUIRE(!value.attribute_values.has_value());
    REQUIRE(value.data.value() == "blabla");
}

TEST_CASE("abbadingo symbol list parser: simple", "[parsing]") {
    auto input = lexy::zstring_input("1 2 3");
    auto result = lexy::parse<symbol_grammar::symbol_list>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    const auto& value = result.value();
    REQUIRE(value.size() == 3);
}

TEST_CASE("abbadingo trace parser", "[parsing]") {
    auto input = lexy::zstring_input("0 3 1 2 3");
    auto result = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.label == "0");
    REQUIRE(value.trace_info.number == 3);
    REQUIRE(value.symbols.size() == 3);
    REQUIRE(value.symbols.at(0).name == "1");
    REQUIRE(value.symbols.at(1).name == "2");
    REQUIRE(value.symbols.at(2).name == "3");
}

TEST_CASE("abbadingo trace parser 2", "[parsing]") {
    auto input = lexy::zstring_input("label 3 a b c");
    auto result = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.label == "label");
    REQUIRE(value.trace_info.number == 3);
    REQUIRE(value.symbols.size() == 3);
    REQUIRE(value.symbols.at(0).name == "a");
    REQUIRE(value.symbols.at(1).name == "b");
    REQUIRE(value.symbols.at(2).name == "c");
}

TEST_CASE("abbadingo trace parser 3", "[parsing]") {
    auto input = lexy::zstring_input("label 3:1.0/bar a:2.0 b:3.0 c:4.0/foo");
    auto result = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.label == "label");
    REQUIRE(value.trace_info.number == 3);
    REQUIRE(value.trace_info.attribute_values.value().at(0) == "1.0");
    REQUIRE(value.trace_info.data.value() == "bar");
    REQUIRE(value.symbols.size() == 3);
    REQUIRE(value.symbols.at(0).name == "a");
    REQUIRE(value.symbols.at(0).attribute_values.value().at(0) == "2.0");
    REQUIRE(value.symbols.at(1).name == "b");
    REQUIRE(value.symbols.at(1).attribute_values.value().at(0) == "3.0");
    REQUIRE(value.symbols.at(2).name == "c");
    REQUIRE(value.symbols.at(2).attribute_values.value().at(0) == "4.0");
    REQUIRE(value.symbols.at(2).data.value() == "foo");
}

TEST_CASE("abbadingo trace parser 4", "[parsing]") {
    auto input = lexy::zstring_input("0 2 a:1.0 b:2.0");
    auto result = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();

    REQUIRE(value.symbols.at(0).name == "a");
    REQUIRE(value.symbols.at(0).attribute_values.value().at(0) == "1.0");

    REQUIRE(value.symbols.at(1).name == "b");
    REQUIRE(value.symbols.at(1).attribute_values.value().at(0) == "2.0");
}

TEST_CASE("abbadingo trace parser: empty trace", "[parsing]") {
    auto input = lexy::zstring_input("0 0");
    auto result = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.symbols.empty());
}

TEST_CASE("abbadingo trace parser: error incomplete attributes", "[parsing]") {
    auto input = lexy::zstring_input("0 1 a:");
    auto result = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);
    REQUIRE(!result.has_value());
}

TEST_CASE("abbadingo trace parser: error incomplete data", "[parsing]") {
    auto input = lexy::zstring_input("0 1 a/");
    auto result = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);
    REQUIRE(!result.has_value());
}

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

TEST_CASE("abbadingo_parser: smoke test with trace attributes", "[parsing]") {
    std::string input = "2:ds/tatt1,ft/tatt2 2\n"
                        "0 2:1.0,2.0 a b\n"
                        "1 2:3.0,4.0 c d";
    std::istringstream inputstream(input);

    auto parser = abbadingoparser(inputstream);

    auto first = parser.next().value();
    auto first_trace_attr_info = first.get_trace_attr_info();
    REQUIRE(first.get("id") == std::vector<std::string> {"0"});
    REQUIRE(first.get("symb") == std::vector<std::string> {"a"});
    REQUIRE(first.get("type") == std::vector<std::string> {"0"});


    auto second = parser.next().value();
    auto second_trace_attr_info = second.get_trace_attr_info();
    REQUIRE(second.get("id") == std::vector<std::string> {"0"});
    REQUIRE(second.get("symb") == std::vector<std::string> {"b"});
    REQUIRE(second.get("type") == std::vector<std::string> {"0"});

    REQUIRE(first_trace_attr_info == second_trace_attr_info);

    auto third = parser.next().value();
    auto third_trace_attr_info = third.get_trace_attr_info();
    REQUIRE(third.get("id") == std::vector<std::string> {"1"});
    REQUIRE(third.get("symb") == std::vector<std::string> {"c"});
    REQUIRE(third.get("type") == std::vector<std::string> {"1"});

    auto fourth = parser.next().value();
    auto fourth_trace_attr_info = fourth.get_trace_attr_info();
    REQUIRE(fourth.get("id") == std::vector<std::string> {"1"});
    REQUIRE(fourth.get("symb") == std::vector<std::string> {"d"});
    REQUIRE(fourth.get("type") == std::vector<std::string> {"1"});

    REQUIRE(third_trace_attr_info == fourth_trace_attr_info);

    REQUIRE(first_trace_attr_info->at(0).get_name() == "tatt1");
    REQUIRE(first_trace_attr_info->at(1).get_name() == "tatt2");
    REQUIRE(first_trace_attr_info->at(0).get_value() == "1.0");
    REQUIRE(first_trace_attr_info->at(1).get_value() == "2.0");

    REQUIRE(first_trace_attr_info->at(0).is_discrete());
    REQUIRE(first_trace_attr_info->at(0).is_splittable());
    REQUIRE(!first_trace_attr_info->at(0).is_distributionable());
    REQUIRE(!first_trace_attr_info->at(0).is_target());
    REQUIRE(!first_trace_attr_info->at(1).is_discrete());
    REQUIRE(!first_trace_attr_info->at(1).is_splittable());
    REQUIRE(first_trace_attr_info->at(1).is_distributionable());
    REQUIRE(first_trace_attr_info->at(1).is_target());

    REQUIRE(third_trace_attr_info->at(0).get_name() == "tatt1");
    REQUIRE(third_trace_attr_info->at(1).get_name() == "tatt2");
    REQUIRE(third_trace_attr_info->at(0).get_value() == "3.0");
    REQUIRE(third_trace_attr_info->at(1).get_value() == "4.0");

    REQUIRE(third_trace_attr_info->at(0).is_discrete());
    REQUIRE(third_trace_attr_info->at(0).is_splittable());
    REQUIRE(!third_trace_attr_info->at(0).is_distributionable());
    REQUIRE(!third_trace_attr_info->at(0).is_target());
    REQUIRE(!third_trace_attr_info->at(1).is_discrete());
    REQUIRE(!third_trace_attr_info->at(1).is_splittable());
    REQUIRE(third_trace_attr_info->at(1).is_distributionable());
    REQUIRE(third_trace_attr_info->at(1).is_target());
}


TEST_CASE("abbadingo_parser: smoke test with symbol attributes", "[parsing]") {
    std::string input = "2 2:d/sattr1\n"
                        "0 2 a:1.0 b:2.0\n"
                        "1 2 c:3.0 d:4.0";
    std::istringstream inputstream(input);

    auto parser = abbadingoparser(inputstream);

    auto first = parser.next().value();
    auto first_sattr_info = first.get_symb_attr_info();
    REQUIRE(first.get("id") == std::vector<std::string> {"0"});
    REQUIRE(first.get("symb") == std::vector<std::string> {"a"});
    REQUIRE(first.get("type") == std::vector<std::string> {"0"});
    REQUIRE(first_sattr_info.size() == 1);
    REQUIRE(first_sattr_info.at(0).get_value() == "1.0");
    REQUIRE(first_sattr_info.at(0).is_discrete());
    REQUIRE(!first_sattr_info.at(0).is_splittable());
    REQUIRE(!first_sattr_info.at(0).is_distributionable());
    REQUIRE(!first_sattr_info.at(0).is_target());

    auto second = parser.next().value();
    auto second_sattr_info = second.get_symb_attr_info();
    REQUIRE(second.get("id") == std::vector<std::string> {"0"});
    REQUIRE(second.get("symb") == std::vector<std::string> {"b"});
    REQUIRE(second.get("type") == std::vector<std::string> {"0"});
    REQUIRE(second_sattr_info.size() == 1);
    REQUIRE(second_sattr_info.at(0).get_value() == "2.0");
    REQUIRE(second_sattr_info.at(0).is_discrete());
    REQUIRE(!second_sattr_info.at(0).is_splittable());
    REQUIRE(!second_sattr_info.at(0).is_distributionable());
    REQUIRE(!second_sattr_info.at(0).is_target());

    auto third = parser.next().value();
    auto third_sattr_info = third.get_symb_attr_info();
    REQUIRE(third.get("id") == std::vector<std::string> {"1"});
    REQUIRE(third.get("symb") == std::vector<std::string> {"c"});
    REQUIRE(third.get("type") == std::vector<std::string> {"1"});
    REQUIRE(third_sattr_info.size() == 1);
    REQUIRE(third_sattr_info.at(0).get_value() == "3.0");
    REQUIRE(third_sattr_info.at(0).is_discrete());
    REQUIRE(!third_sattr_info.at(0).is_splittable());
    REQUIRE(!third_sattr_info.at(0).is_distributionable());
    REQUIRE(!third_sattr_info.at(0).is_target());

    auto fourth = parser.next().value();
    auto fourth_sattr_info = fourth.get_symb_attr_info();
    REQUIRE(fourth.get("id") == std::vector<std::string> {"1"});
    REQUIRE(fourth.get("symb") == std::vector<std::string> {"d"});
    REQUIRE(fourth.get("type") == std::vector<std::string> {"1"});
    REQUIRE(fourth_sattr_info.size() == 1);
    REQUIRE(fourth_sattr_info.at(0).get_value() == "4.0");
    REQUIRE(fourth_sattr_info.at(0).is_discrete());
    REQUIRE(!fourth_sattr_info.at(0).is_splittable());
    REQUIRE(!fourth_sattr_info.at(0).is_distributionable());
    REQUIRE(!fourth_sattr_info.at(0).is_target());

}

TEST_CASE("abbadingo_parser: add single trace", "[parsing]") {
    std::string input = "0 3 a b c";
    std::istringstream inputstream(input);

    auto parser = abbadingoparser::single_trace(inputstream);
    auto input_data = inputdata();
    inputdata_locator::provide(&input_data);
    input_data.read(&parser);

    std::vector<trace*> traces(input_data.begin(), input_data.end());

    REQUIRE(traces.size() == 1);
    REQUIRE(traces.at(0)->to_string() == "0 3 a b c");
}

TEST_CASE("abbadingo trace parser - trailing whitespace", "[parsing]") {
    auto input = lexy::zstring_input("label 3 a b c          ");
    auto result = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);
    REQUIRE(result.has_value());
    auto value = result.value();
    REQUIRE(value.label == "label");
    REQUIRE(value.trace_info.number == 3);
    REQUIRE(value.symbols.size() == 3);
    REQUIRE(value.symbols.at(0).name == "a");
    REQUIRE(value.symbols.at(1).name == "b");
    REQUIRE(value.symbols.at(2).name == "c");
}