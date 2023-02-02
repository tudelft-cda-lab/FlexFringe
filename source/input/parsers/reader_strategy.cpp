//
// Created by tom on 2/2/23.
//

#include "reader_strategy.h"


std::optional<trace *> read_all::read(parser &input_parser, inputdata &idata) {
    if (!has_read) {
        consume_all(input_parser, idata);
        has_read = true;
    }

    if (traces.empty()) {
        return std::nullopt;
    } else {
        auto trace = traces.front();
        traces.pop_front();
        return trace;
    }
}

void read_all::consume_all(parser &input_parser, inputdata &idata) {
    while (true) {
        // Do we have another symbol from the parser?
        auto cur_symbol_maybe = input_parser.next();
        if (!cur_symbol_maybe.has_value()) {
            break;
        }

        // Process it!
        auto cur_symbol = cur_symbol_maybe.value();
        idata.process_symbol_info(cur_symbol, trace_map);
    }

    for (const auto &[id, trace]: trace_map) {
        traces.push_back(trace);
    }
}

std::optional<trace *> sentinel_symbol::read(parser &input_parser, inputdata &idata) {

    while (true) {
        // Do we have another symbol from the parser?
        auto cur_symbol_maybe = input_parser.next();
        if (!cur_symbol_maybe.has_value()) {
            break;
        }
        auto cur_symbol = cur_symbol_maybe.value();

        // If it's a sentinel symbol, we are done with the trace it belongs to
        if (cur_symbol.get_str("symb") == sentinel) {
            return trace_map.at((cur_symbol.get_str("id")));
        }

        // Otherwise, process it!
        idata.process_symbol_info(cur_symbol, trace_map);
    }

    return std::nullopt;
}

std::optional<trace *> in_order::read(parser &input_parser, inputdata &idata) {

    while (true) {
        // Do we have another symbol from the parser?
        auto cur_symbol_maybe = input_parser.next();
        if (!cur_symbol_maybe.has_value()) {
            // We are done with the last trace, if we had any symbols at all
            if (last_id.has_value()) {
                return trace_map.at(last_id.value());
            }
            // Otherwise there was no trace
            break;
        }

        auto cur_symbol = cur_symbol_maybe.value();
        auto cur_id = cur_symbol.get_str("id");

        // Start of the first trace
        if (!last_id.has_value()) {
            last_id = cur_id;
        }

        // Process the incoming symbol
        idata.process_symbol_info(cur_symbol, trace_map);

        // Previous trace finished?
        if (last_id.value() != cur_id) {
            auto ret_id = last_id.value();
            last_id = cur_id;
            return trace_map.at(ret_id);
        }
    }

    return std::nullopt;
}
