//
// Created by tom on 1/6/23.
//

#ifndef FLEXFRINGE_ABBADINGOPARSER_H
#define FLEXFRINGE_ABBADINGOPARSER_H


#include <deque>
#include "i_parser.h"

class abbadingoparser: public parser {
private:
    std::istream& inputstream;
    ssize_t max_sequences;
    ssize_t alphabet_size;

    ssize_t num_lines_processed;
    std::deque<symbol_info> symbols;

public:
    abbadingoparser(std::istream &input)
    : inputstream(input)
    {
        max_sequences = 0;
        alphabet_size = 0;
        num_lines_processed = 0;

        parse_header();
    }

    void parse_header();

    bool read_abbadingo_trace();

    std::optional<symbol_info> next();
};


#endif //FLEXFRINGE_ABBADINGOPARSER_H
