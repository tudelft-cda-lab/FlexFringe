//
// Created by tom on 1/6/23.
//



#include "input/parsers/abbadingoparser.h"
#include "stringutil.h"
#include "input/parsers/grammar/abbadingoheader.h"
#include "input/parsers/grammar/abbadingosymbol.h"


#include <lexy/action/parse.hpp>
#include <lexy_ext/report_error.hpp>
#include <lexy/input/string_input.hpp>
#include <fmt/format.h>

#include <sstream>
#include <chrono>

void abbadingoparser::parse_header(std::istream &inputstream) {
    std::string line;
    std::getline(inputstream, line);

    auto input = lexy::string_input(line);
    auto parsed_header_maybe = lexy::parse<grammar::abbadingo_header_p>(input, lexy_ext::report_error);

    if (!parsed_header_maybe.has_value()) {
        throw std::runtime_error("Could not parse abbadingo header");
    }

    auto parsed_header = parsed_header_maybe.value();

    // Keep a copy of the parsed header data around
    header_info = std::make_unique<abbadingo_header_info>(parsed_header);

    // Create prototypes for the trace and symbol attribute info data
    for (auto &tattr_info: parsed_header.traces.attributes) {
        trace_attr_prototypes.emplace_back(tattr_info.name,
                                           "",
                                           tattr_info.type);
    }
    for (auto &attr_info: parsed_header.symbols.attributes) {
        symbol_attr_prototypes.emplace_back(attr_info.name,
                                            "",
                                            attr_info.type);
    }
}

bool abbadingoparser::read_abbadingo_trace() {

    std::string line;
    if (!std::getline(inputstream, line)) {
        return false;
    }

    auto line_idx = num_lines_processed + 2;

    auto input = lexy::string_input(line);
    auto parsed_trace_maybe = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);

    // Did we parse successfully?
    if (!parsed_trace_maybe.has_value()) {
        throw std::runtime_error(fmt::format("Error parsing abbadingo input: line {}", line_idx));
    }
    auto trace = parsed_trace_maybe.value();

    // Is the specified amount of symbols in the trace equal to the actual amount?
    if (trace.trace_info.number != trace.symbols.size()) {
        throw std::runtime_error(
                fmt::format("Error parsing abbadingo input: line {} - Incorrectly specified number of symbols in trace",
                            line_idx));
    }

    // Gather trace attribute info
    // We use the prototype attribute_info objects and fill in the values accordingly
    auto trace_attribute_info = std::make_shared<std::vector<attribute_info>>();
    if (trace.trace_info.attribute_values.has_value()) {
        auto trace_attribute_values = trace.trace_info.attribute_values.value();
        auto num_tattr_values = trace_attribute_values.size();
        auto expected_num_tattr_values = trace_attr_prototypes.size();

        if (num_tattr_values != expected_num_tattr_values) {
            throw std::runtime_error(
                    fmt::format("Error parsing abbadingo input: line {} - expected {} trace attributes, found {}",
                                line_idx, expected_num_tattr_values, num_tattr_values));
        }

        for (size_t i = 0; i < num_tattr_values; i++) {
            auto cur_prototype = trace_attr_prototypes.at(i);
            auto cur_value = trace_attribute_values.at(i);
            trace_attribute_info->push_back(cur_prototype.clone(std::string(cur_value)));
        }
    }

    // Construct the symbol info for this trace

    // Handle special case of empty trace (it can still have useful information even without symbols)
    // we do this by specifying a symbol without a "symb" value defined
    if (trace.symbols.empty()) {
        symbol_info cur_symbol;

        cur_symbol.set("id", std::to_string(num_lines_processed));
        cur_symbol.set("type", std::string{trace.label});

        // No symbol attributes possible
        // But we can have trace attributes
        cur_symbol.set_trace_attr_info(trace_attribute_info);
        symbols.push_back(cur_symbol);
    }
    // If we do have symbols, just grab all the info we have and stuff it in a symbolinfo
    else {
        for (const auto &symbol: trace.symbols) {
            symbol_info cur_symbol;

            cur_symbol.set("id", std::to_string(num_lines_processed));
            cur_symbol.set("symb", std::string{symbol.name});
            cur_symbol.set("type", std::string{trace.label}); // Not sure if this is the correct place to put this

            // Construct the symbol attribute info objects if we have any
            if (symbol.attribute_values.has_value()) {
                auto symbol_attr_vals = symbol.attribute_values.value();
                auto num_sattr_values = symbol_attr_vals.size();
                auto expected_num_sattr_values = symbol_attr_prototypes.size();

                if (num_sattr_values != expected_num_sattr_values) {
                    throw std::runtime_error(
                            fmt::format(
                                    "Error parsing abbadingo input: line {} - expected {} symbol attributes, found {}",
                                    line_idx, num_sattr_values, expected_num_sattr_values));
                }

                for (size_t i = 0; i < num_sattr_values; i++) {
                    auto cur_prototype = symbol_attr_prototypes.at(i);
                    auto cur_value = symbol_attr_vals.at(i);
                    cur_symbol.push_symb_attr_info(cur_prototype.clone(std::string(cur_value)));
                }
            }

            cur_symbol.set_trace_attr_info(trace_attribute_info);

            if (symbol.data.has_value()) {
                cur_symbol.set("eval", std::string(symbol.data.value()));
            }

            symbols.push_back(cur_symbol);
        }
    }

    num_lines_processed++;

    return true;
}

std::optional<symbol_info> abbadingoparser::next() {
    // If we don't have any new symbols available, read the next trace
    // If there are no new traces to read, we are done
    while (symbols.empty()) {
        if (!read_abbadingo_trace()) {
            return std::nullopt;
        }
    }

    // Grab the next symbol and return it
    symbol_info cur_symbol = symbols.front();
    symbols.pop_front();
    return cur_symbol;
}

/**
 * Helper method to read a single abbadingo formatted trace
 * Currently does not support trace or symbol attributes
 *
 * @param trace an input stream containing an abbadingo formatted trace.
 * @return a parser for the trace.
 */
abbadingoparser abbadingoparser::single_trace(std::istream &trace) {

    abbadingoparser parser(trace, false);

    std::stringstream header;
    header << "1 0\n";
    parser.parse_header(header);

    return parser;
}
