//
// Created by tom on 1/5/23.
//

#ifndef FLEXFRINGE_I_PARSER_H
#define FLEXFRINGE_I_PARSER_H

#include <vector>
#include "input/trace.h"

class parser {
    virtual void parse() = 0;
    virtual std::vector<trace*> get_traces() = 0;
};

#endif //FLEXFRINGE_I_PARSER_H
