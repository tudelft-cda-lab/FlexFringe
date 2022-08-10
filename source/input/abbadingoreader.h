#ifndef FLEXFRINGE_ABBADINGOREADER_H
#define FLEXFRINGE_ABBADINGOREADER_H


#include "input/i_inputdata.h"

class AbbadingoInputData: public IInputData {
private:
    void read_abbadingo_header(std::istream &input_stream);
    void read_abbadingo_sequence(std::istream &input_stream, Trace*);
    void read_abbadingo_type(std::istream &input_stream, Trace*);
    void read_abbadingo_symbol(std::istream &input_stream, Tail*);
public:
    void read(std::istream &input_stream);
};


#endif //FLEXFRINGE_ABBADINGOREADER_H
