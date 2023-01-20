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

void abbadingoparser::parse_header() {
    std::string line;
    std::getline(inputstream, line);

    auto input = lexy::string_input(line);
    auto parsed_header_maybe = lexy::parse<grammar::abbadingo_header_p>(input, lexy_ext::report_error);

    if (!parsed_header_maybe.has_value()) {
        throw std::runtime_error("Could not parse abbadingo header");
    }

    auto parsed_header = parsed_header_maybe.value();
    max_sequences = parsed_header.traces.number;
    alphabet_size = parsed_header.symbols.number;

    //TODO: attributes etc.
}

bool abbadingoparser::read_abbadingo_trace() {


    std::string line;
    if (!std::getline(inputstream, line)) {
        return false;
    }

    auto line_idx = num_lines_processed + 2;

//    auto before = std::chrono::high_resolution_clock::now();
    auto input = lexy::string_input(line);
    auto parsed_trace_maybe = lexy::parse<symbol_grammar::abbadingo_trace>(input, lexy_ext::report_error);
//    auto after = std::chrono::high_resolution_clock::now();

    // Did we parse successfully?
    if (!parsed_trace_maybe.has_value()) {
        throw std::runtime_error(fmt::format("Error parsing abbadingo input: line {}", line_idx));
    }
    auto trace = parsed_trace_maybe.value();

    // Is the specified amount of symbols in the trace equal to the actual amount?
    if (trace.trace_info.number != trace.symbols.size()) {
        throw std::runtime_error(fmt::format("Error parsing abbadingo input: line {} - Incorrectly specified number of symbols in trace", line_idx));
    }

    for (const auto &symbol: trace.symbols) {
        symbol_info cur_symbol;

        cur_symbol.set("id", std::to_string(num_lines_processed) );
        cur_symbol.set("symb", std::string {symbol.name});
        cur_symbol.set("type", std::string {trace.label}); // Not sure if this is the correct place to put this

        // TODO: attributes & data

        symbols.push_back(cur_symbol);
    }

    num_lines_processed++;

//    auto ms = std::chrono::duration<double, std::milli>(after - before);
//    std::cout << "Parsing trace took " << ms.count() << "ms" << "\n";
    return true;
}

std::optional<symbol_info> abbadingoparser::next() {
    // If we don't have any new symbols available, read the next trace
    // If there are no new traces to read, we are done
    while (symbols.empty()) {
        if(!read_abbadingo_trace()) {
            return std::nullopt;
        }
    }

    // Grab the next symbol and return it
    symbol_info cur_symbol = symbols.front();
    symbols.pop_front();
    return cur_symbol;
}
