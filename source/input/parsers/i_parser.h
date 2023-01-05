//
// Created by tom on 1/5/23.
//

#ifndef FLEXFRINGE_I_PARSER_H
#define FLEXFRINGE_I_PARSER_H

#include "input/inputdata.h"
#include <vector>
#include "input/trace.h"


class parser {
public:
    virtual void parse(inputdata *pInputdata) = 0;
    virtual std::vector<trace*> get_traces() = 0;
};

#endif //FLEXFRINGE_I_PARSER_H
