//
// Created by tom on 1/30/23.
//

#ifndef FLEXFRINGE_STREAMING_H
#define FLEXFRINGE_STREAMING_H

#include "input/inputdata.h"

class streaminginput {
private:
    inputdata &input_data;
public:
    streaminginput(inputdata& input_data)
            : input_data(input_data)
    {}


};


#endif //FLEXFRINGE_STREAMING_H
