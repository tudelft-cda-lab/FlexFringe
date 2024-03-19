//
// Created by tom on 1/6/23.
//

#ifndef FLEXFRINGE_ABBADINGOPARSER_H
#define FLEXFRINGE_ABBADINGOPARSER_H


#include <deque>
#include "i_parser.h"
#include "input/parsers/grammar/abbadingoheader.h"

class abbadingoparser: public parser {
private:
    std::istream& inputstream;

    std::unique_ptr<abbadingo_header_info> header_info;

    size_t num_lines_processed {};
    std::deque<symbol_info> symbols;

    std::vector<attribute_info> trace_attr_prototypes;
    std::vector<attribute_info> symbol_attr_prototypes;



public:
    explicit abbadingoparser(std::istream &input, bool header = true)
    : inputstream(input)
    {
        if (header) {
            parse_header(inputstream);
        }
    }

    void parse_header(std::istream &inputstream);

    bool read_abbadingo_trace();

    std::optional<symbol_info> next();

    static abbadingoparser single_trace(std::istream &input);
};


#endif //FLEXFRINGE_ABBADINGOPARSER_H
