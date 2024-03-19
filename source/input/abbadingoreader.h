#ifndef FLEXFRINGE_ABBADINGOREADER_H
#define FLEXFRINGE_ABBADINGOREADER_H


#include "input/inputdata.h"

// Left in for compatibility, Do not use
// TODO: remove usages and delete
class abbadingo_inputdata: public inputdata {
public:
    [[deprecated]]
    void read(std::istream &input_stream);
    [[deprecated]]
    void read_abbadingo_header(std::istream &input_stream);
    [[deprecated]]
    void read_abbadingo_sequence(std::istream &input_stream, trace*);
    [[deprecated]]
    void read_abbadingo_type(std::istream &input_stream, trace*);
    [[deprecated]]
    void read_abbadingo_symbol(std::istream &input_stream, tail*);
};


#endif //FLEXFRINGE_ABBADINGOREADER_H
