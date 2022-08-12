#ifndef FLEXFRINGE_IREADER_H
#define FLEXFRINGE_IREADER_H

#include <list>
#include <vector>
#include <map>
#include <istream>
//#include "inputdata.h"
#include "input/trace.h"
#include "input/attribute.h"

class apta;

class inputdata {
protected:
    std::list<trace*> traces;

    std::vector<std::string> alphabet;
    std::map<std::string, int> r_alphabet;

    std::vector<std::string> types;
    std::map<std::string, int> r_types;

    std::vector<attribute> trace_attributes;
    std::vector<attribute> symbol_attributes;

    int max_sequences;
    int num_sequences;
    int node_number;
    int num_tails;


public:
    using Iterator = std::list<trace*>::iterator;

    virtual void read(std::istream &input_stream) = 0;

    void add_traces_to_apta(apta *the_apta);
    void add_trace_to_apta(trace *tr, apta *the_apta);

    std::string& get_symbol(int a);
    inline int get_reverse_symbol(std::string a);
    inline std::string& get_type(int a);
    inline int get_reverse_type(std::string a);

    /* gets an attribute, first symbol attributes, then trace attributes */
    inline attribute* get_trace_attribute(int attr);
    inline attribute* get_symbol_attribute(int attr);
    inline attribute* get_attribute(int attr);

    inline int get_num_symbol_attributes();
    inline int get_num_trace_attributes();
    inline int get_num_attributes();

    /* attribute properties:
     * splittable: will be used to infer guards
     * distributionable: will be used in evaluation functions that model attributes
     * discrete: whether the attribute discrete or continuous
     * target: will be used in evaluation functions as class/target/prediction variable
     * */
    inline bool is_splittable(int attr);
    inline bool is_distributionable(int attr);
    inline bool is_discrete(int attr);
    inline bool is_target(int attr);

    /* inputdata properties:
     * the number of distinct sequence types
     * the size of the input data
     * */
    inline int get_types_size();
    inline int get_alphabet_size();

    inline int symbol_from_string(std::string symbol);
    inline std::string string_from_symbol(int symbol);
    inline int type_from_string(std::string type);
    std::string string_from_type(int type);

    trace* access_trace(tail *t);
    tail* access_tail(tail *t);

    Iterator begin() {return traces.begin();}
    Iterator end() {return traces.end();}
};

#endif //FLEXFRINGE_IREADER_H
