//
// Created by tom on 2/2/23.
//

#ifndef FLEXFRINGE_READER_STRATEGY_H
#define FLEXFRINGE_READER_STRATEGY_H

#include <vector>
#include <optional>
#include "input/parsers/i_parser.h"
#include "input/inputdata.h"
#include "input/trace.h"

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif


class parser;

class reader_strategy {
public:
    virtual std::optional<trace*> read(parser& input_parser, inputdata& idata) = 0;
};

/**
 * This reader strategy just consumes all the available input data
 * and builds traces with them, assuming that all traces are done when
 * no more input data is available.
 */
class read_all: public reader_strategy {
private:
    std::unordered_map<std::string, trace *> trace_map;
    std::list<trace*> traces;
    bool has_read = false;
    void consume_all(parser& input_parser, inputdata& idata);

public:
    std::optional<trace*> read(parser& input_parser, inputdata& idata) override;
};

class slidingwindow: public reader_strategy {
private:
    std::unordered_map<std::string, trace *> trace_map;
    ssize_t sliding_window_size {};
    ssize_t sliding_window_stride {};
    bool sliding_window_type {};
    int num_sequences {};

public:
    slidingwindow(ssize_t sliding_window_size,
                  ssize_t sliding_window_stride,
                  bool sliding_window_type);
    std::optional<trace*> read(parser& input_parser, inputdata& idata) override;
};

/**
 * This reader strategy reads traces that are delimited by a specified sentinel symbol
 * e.g. a->b->c->x with sentinel symbol x would result in the trace a->b->c
 */
class sentinel_symbol: public reader_strategy {
private:
    std::unordered_map<std::string, trace *> trace_map;
    std::string sentinel;

public:
    explicit sentinel_symbol(std::string symbol): sentinel(std::move(symbol)) {}
    std::optional<trace*> read(parser& input_parser, inputdata& idata) override;
};

/**
 * This strategy assumes all symbols for one trace arrive after each other.
 * A trace is finished once a symbol belonging to a different trace arrives.
 * Mostly useful with abbaddingo input data.
 */
class in_order: public reader_strategy {
private:
    std::unordered_map<std::string, trace *> trace_map;
    std::optional<std::string> last_id {};

public:
    std::optional<trace*> read(parser& input_parser, inputdata& idata) override;
};
#endif //FLEXFRINGE_READER_STRATEGY_H
