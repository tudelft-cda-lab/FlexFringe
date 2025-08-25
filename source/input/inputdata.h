#ifndef FLEXFRINGE_INPUTDATA_H
#define FLEXFRINGE_INPUTDATA_H

#include <list>
#include <vector>
#include <map>
#include <istream>
#include <memory>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <istream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <optional>
#include <string>

#include "input/trace.h"
#include "input/attribute.h"
#include "input/parsers/i_parser.h"
#include "input/parsers/reader_strategy.h"

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

class apta;
class csv_parser;
class parser;
class symbol_info;
class reader_strategy;
class TraceIterator;

class inputdata {
protected:
    std::unique_ptr<parser> parser_;

    std::list<trace*> traces;

    std::vector<std::string> alphabet;
    std::unordered_map<std::string, int> r_alphabet;

    std::vector<std::string> types;
    std::unordered_map<std::string, int> r_types;

    std::vector<attribute> trace_attributes;
    std::vector<attribute> symbol_attributes;

    int max_sequences {};
    int num_sequences {};
    int node_number {};
    int num_tails {};

    tail *make_tail(const std::string &symbol,
                    const std::vector<std::string> &data);

    void add_type_to_trace(trace *new_trace,
                           const std::string &type);
private:

    // The trace IDs of which we already processed the trace attributes
    std::set<std::string> processed_trace_ids {}; // TODO: should we use string_view and unordered_set here?

    void process_trace_attributes(symbol_info &symbolinfo, trace* tr);
    void process_symbol_attributes(symbol_info &symbolinfo, tail* t);

public:
    using Iterator = std::list<trace*>::iterator;


    void read(parser* input_parser);
    void read_slidingwindow(parser* input_parser,
                            ssize_t sliding_window_size     = 10,
                            ssize_t sliding_window_stride   = 1,
                            bool sliding_window_type        = false);

    std::optional<trace*> read_trace(parser& input_parser, reader_strategy& strategy, bool save = true);

    std::pair<trace*, tail*> process_symbol_info(symbol_info &symbolinfo,
                                                 std::unordered_map<std::string, trace*> &trace_map);

    static inputdata with_alphabet_from(inputdata& other);

    void add_traces_to_apta(apta *the_apta, const bool use_thresholds=true);
    void add_trace_to_apta(trace *tr, apta *the_apta, const bool use_thresholds=true, 
                           std::unordered_set<int>* states_to_append_to=nullptr);

    std::string& get_symbol(int a);
    int get_reverse_symbol(std::string a);
    const std::string& get_type(int a);
    int get_reverse_type(std::string a);
    void set_alphabet(const std::vector<std::string>& input_alphabet);
    void set_alphabet(const std::unordered_map<std::string, int>& input_r_alphabet);
    void set_types(const std::vector<std::string>& input_r_types);
    void set_types(const std::unordered_map<std::string, int>& input_r_types);

    /* gets an attribute, first symbol attributes, then trace attributes */
    attribute* get_trace_attribute(int attr);
    attribute* get_symbol_attribute(int attr);
    attribute* get_attribute(int attr);

    int get_num_symbol_attributes();
    int get_num_trace_attributes();
    int get_num_attributes();
    int get_num_sequences();
    int get_max_sequences();

    void add_type(const std::string& t);
    const std::vector<int> get_types() const;
    const std::unordered_map<std::string, int>& get_r_types() const;

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
    const std::vector<int> get_alphabet() const;
    const std::unordered_map<std::string, int> get_r_alphabet() const;

    int symbol_from_string(std::string symbol);

    std::string string_from_symbol(int symbol);
    int type_from_string(std::string type);
    std::string string_from_type(int type);

    trace* access_trace(tail *t);
    tail* access_tail(tail *t);

    void add_trace(trace* tr) noexcept;

    Iterator begin() {return traces.begin();}
    Iterator end() {return traces.end();}

    void clear_traces() noexcept{this->traces.clear();}

    /**
     * @brief Loop over traces using read_trace in sequential fashion without saving the traces in a vector.
     * Erase the trace afterwards to preserve memory using this iterator.
     */
    TraceIterator trace_iterator(parser& input_parser, reader_strategy& strategy);
};

class TraceIterator {
    inputdata& idat;
    parser& input_parser;
    reader_strategy& strategy;

    class iterator {
        inputdata* idat;
        parser* input_parser;
        reader_strategy* strategy;
        trace* current;

        public:
        iterator(inputdata* idat, parser* input_parser, reader_strategy* strategy) : idat(idat), input_parser(input_parser), strategy(strategy){
            increment();
        }
        explicit iterator(trace* t = nullptr) : current(t) {}
        iterator& operator++() {
            increment();
            return *this;
        }
        trace* operator*() {return current;}
        void increment() {
            bool save = false; // do not save traces to reduce memory consumption.
            std::optional<trace*> trace_opt = idat->read_trace(*input_parser, *strategy, save);
            if (!trace_opt) { current = nullptr; }
            else { current = trace_opt.value(); }
        }
        bool operator==(iterator other) const {return this->current == other.current;}
        bool operator!=(iterator other) const {return !(*this == other);}
    };
    public:
    TraceIterator(inputdata& idat, parser& input_parser, reader_strategy& strategy) : idat(idat), input_parser(input_parser), strategy(strategy){}
    iterator begin() {return iterator{&idat, &input_parser, &strategy};}
    iterator end() {return iterator();}
};

#endif //FLEXFRINGE_IREADER_H
