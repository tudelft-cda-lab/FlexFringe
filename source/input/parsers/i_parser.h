//
// Created by tom on 1/5/23.
//

#ifndef FLEXFRINGE_I_PARSER_H
#define FLEXFRINGE_I_PARSER_H

#include "input/inputdata.h"
#include "input/trace.h"
#include "input/parsers/symbol_info.h"
#include <vector>
#include <optional>


class parser {
public:
    virtual std::optional<symbol_info> next() = 0;
    virtual ~parser() = default;
};

#endif //FLEXFRINGE_I_PARSER_H
