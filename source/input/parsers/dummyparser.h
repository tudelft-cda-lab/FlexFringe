//
// Created by tom on 1/30/23.
//

#ifndef FLEXFRINGE_DUMMYPARSER_H
#define FLEXFRINGE_DUMMYPARSER_H

#include <deque>
#include <optional>

#include "input/parsers/symbol_info.h"
#include "input/parsers/i_parser.h"

class dummyparser: public parser {
private:
    std::deque<symbol_info> symbols;

public:
    dummyparser& add(const symbol_info& s);
    std::optional<symbol_info> next() override;
};


#endif //FLEXFRINGE_DUMMYPARSER_H
