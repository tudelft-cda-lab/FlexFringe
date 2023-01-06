#ifndef FLEXFRINGE_INPUTDATA_H
#define FLEXFRINGE_INPUTDATA_H

#include <list>
#include <vector>
#include <map>
#include <istream>
#include <memory>
#include "input/trace.h"
#include "input/attribute.h"
#include "input/parsers/i_parser.h"

class apta;
class csv_parser;
class parser;

class inputdata {
    friend class csv_parser;
protected:
    std::unique_ptr<parser> parser_;

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

//    inputdata(std::unique_ptr<parser> parser)
//    : parser_(std::move(parser))
//    {
////        parser_->parse(this);
//    }
//stub
    virtual void read(std::istream &input_stream) = 0;

    void add_traces_to_apta(apta *the_apta);
    void add_trace_to_apta(trace *tr, apta *the_apta);

    std::string& get_symbol(int a);
    int get_reverse_symbol(std::string a);
    std::string& get_type(int a);
    int get_reverse_type(std::string a);

    /* gets an attribute, first symbol attributes, then trace attributes */
    attribute* get_trace_attribute(int attr);
    attribute* get_symbol_attribute(int attr);
    attribute* get_attribute(int attr);

    int get_num_symbol_attributes();
    int get_num_trace_attributes();
    int get_num_attributes();
    int get_num_sequences();
    int get_max_sequences();

    /* attribute properties:
     * splittable: will be used to infer guards
     * distributionable: will be used in evaluation functions that model attributes
     * discrete: whether the attribute discrete or continuous
     * target: will be used in evaluation functions as class/target/prediction variable
     * */
    bool is_splittable(int attr);
    bool is_distributionable(int attr);
    bool is_discrete(int attr);
    bool is_target(int attr);

    /* inputdata properties:
     * the number of distinct sequence types
     * the size of the input data
     * */
    int get_types_size();
    int get_alphabet_size();

    int symbol_from_string(std::string symbol);

    virtual std::string string_from_symbol(int symbol);
    int type_from_string(std::string type);
    std::string string_from_type(int type);

    trace* access_trace(tail *t);
    tail* access_tail(tail *t);

    Iterator begin() {return traces.begin();}
    Iterator end() {return traces.end();}
};

#endif //FLEXFRINGE_IREADER_H
