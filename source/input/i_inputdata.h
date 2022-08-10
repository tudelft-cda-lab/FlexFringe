#ifndef FLEXFRINGE_IREADER_H
#define FLEXFRINGE_IREADER_H

#include <list>
#include <vector>
#include <map>
#include <istream>
//#include "inputdata.h"
#include "input/trace.h"
#include "input/attribute.h"

class IInputData {
protected:
    std::list<Trace*> traces;

    std::vector<std::string> alphabet;
    std::map<std::string, int> r_alphabet;

    std::vector<std::string> types;
    std::map<std::string, int> r_types;

    std::vector<Attribute> trace_attributes;
    std::vector<Attribute> symbol_attributes;

    int max_sequences;
    int num_sequences;
    int node_number;
    int num_tails;


public:
    virtual void read(std::istream &input_stream) = 0;

    inline int get_num_symbol_attributes() {
        return symbol_attributes.size();
    }
    inline int get_num_trace_attributes() {
        return trace_attributes.size();
    }
    inline std::string& get_symbol(int a){
        return alphabet[a];
    }
    inline std::string string_from_type(int type){
        return types[type];
    }
};

#endif //FLEXFRINGE_IREADER_H
