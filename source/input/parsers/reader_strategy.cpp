//
// Created by tom on 2/2/23.
//

#include "reader_strategy.h"
#include "mem_store.h"

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
                auto trace = trace_map.at(last_id.value());
                trace_map.clear();
                // Remove last_id to prevent returning the last trace infinitely
                last_id = std::nullopt;
                return trace;
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
            auto cur_trace = trace_map.at(ret_id);
            trace_map.erase(ret_id);
            return cur_trace;
        }
    }

    return std::nullopt;
}

slidingwindow::slidingwindow(ssize_t sliding_window_size, ssize_t sliding_window_stride, bool sliding_window_type) {
    if (sliding_window_stride > sliding_window_size) {
        throw std::logic_error("Sliding window stride cannot be bigger than sliding window size");
        // Due to the way it is currently implemented. :(
    }

    if (sliding_window_stride < 1) {
        throw std::logic_error("Sliding window stride cannot be < 1");
    }

    this->sliding_window_size = sliding_window_size;
    this->sliding_window_stride = sliding_window_stride;
    this->sliding_window_type = sliding_window_type;
}

std::optional<trace *> slidingwindow::read(parser &input_parser, inputdata &idata) {
    while (true) {
        auto cur_symbol_maybe = input_parser.next();
        if (!cur_symbol_maybe.has_value()) {
            return std::nullopt;
        }

        auto cur_symbol = cur_symbol_maybe.value();

        auto [tr, new_tail] = idata.process_symbol_info(cur_symbol, trace_map);

        // Sliding window stuff.
        // Definitely some more refactoring potential in here
        if (tr->get_length() == sliding_window_size) {

            // Do some weird type stuff? TODO: what is happening here?
            if (sliding_window_type) {
                std::string type_string = idata.string_from_symbol(new_tail->get_symbol());
                tr->type = idata.type_from_string(type_string);
            }

            // Build the new window trace
            // TODO: also copy over trace and symbol attributes to the new trace & tails.
            trace *new_window = mem_store::create_trace();
            new_window->type = tr->type;
            new_window->sequence = num_sequences++;
            new_window->trace_attr = tr->trace_attr;
            tail *t = tr->get_head();
            tail *new_window_tail = nullptr;
            while (t != nullptr) {
                if (new_window_tail == nullptr) {
                    new_window_tail = mem_store::create_tail(nullptr);
                    new_window->head = new_window_tail;
                    new_window->end_tail = new_window_tail;
                    new_window->length = 1;
                } else {
                    tail *old_tail = new_window_tail;
                    new_window_tail = mem_store::create_tail(nullptr);
                    old_tail->set_future(new_window_tail);
                    new_window->length++;
                    new_window->end_tail = new_window_tail;
                }
                new_window_tail->tr = new_window;
                new_window_tail->td = t->td;
                new_window_tail->split_from = t;
                t = t->future();
            }

            // Chomp up the front of the trace we are sliding over
            for (ssize_t i = 0; i < sliding_window_stride; i++) {
                tr->pop_front();
            }

            return new_window;
        }
    }
}


