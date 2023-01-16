//
// Created by tom on 1/6/23.
//



#include "input/parsers/abbadingoparser.h"
#include "stringutil.h"
#include "input/parsers/grammar/abbadingoheader.h"

#include <lexy/action/parse.hpp>
#include <lexy_ext/report_error.hpp>
#include <lexy/input/string_input.hpp>

#include <sstream>

void abbadingoparser::parse_header() {
    std::string line;
    std::getline(inputstream, line);

    auto input = lexy::string_input(line);
    auto parsed_header_maybe = lexy::parse<grammar::abbadingo_header_p>(input, lexy_ext::report_error);

    if (!parsed_header_maybe.has_value()){
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

    std::istringstream iss(line);

    std::string label, symbol;
    ssize_t len;
    iss >> label >> len;

    while (!iss.eof()) {
        iss >> symbol;
        symbol_info cur_symbol;

        cur_symbol.set("id", std::to_string(num_lines_processed) );
        cur_symbol.set("symb", symbol);
        cur_symbol.set("type", label); // Not sure if this is the correct place to put this

        symbols.push_back(cur_symbol);
    }

    num_lines_processed++;
    return true;
}

std::optional<symbol_info> abbadingoparser::next() {
    // If we don't have any new symbols available, read the next trace
    // If there are no new traces to read, we are done
    if (symbols.empty()) {
        if(!read_abbadingo_trace()) {
            return std::nullopt;
        }
    }

    // Grab the next symbol and return it
    symbol_info cur_symbol = symbols.front();
    symbols.pop_front();
    return cur_symbol;
}
