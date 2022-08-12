
#ifndef _INPUTDATA_H_
#define _INPUTDATA_H_

#include "input/i_inputdata.h"
#include "input/tail.h"
#include "input/trace.h"
#include "input/attribute.h"
#include "input/inputdatalocator.h"


//
//class inputdata;
//class tail;
//class tail_data;
//class trace;
//
//#include <istream>
//#include <sstream>
//#include <iostream>
//#include <string>
//#include <map>
//#include <vector>
//#include <set>
//
//#include "json.hpp"
//#include "mem_store.h"
//
//// for convenience
//using json = nlohmann::json;
//using namespace std;
//
//#include "apta.h"
//#include "mem_store.h"
//
///**
// * @brief Wrapper class for the input data. Supports functionalities
// * such as alphabet functions, file transformations and data added to the APTA.
// *
// */
//class attribute{
//public:
//    bool discrete;
//    bool splittable;
//    bool distributionable;
//    bool target;
//
//    vector<string> values;
//    map<string, int> r_values;
//
//    string name;
//
//    attribute(const string& input);
//
//    inline double get_value(string val){
//        if(discrete){
//            if(r_values.find(val) == r_values.end()) {
//                r_values[val] = values.size();
//                values.push_back(val);
//            }
//            return (double) r_values[val];
//        } else {
//            double result;
//            try {
//                result = std::stof(val);
//            } catch (const std::invalid_argument& e) {
//                result = 0.0;
//            }
//            return result;
//        }
//    };
//
//    inline string get_name(){
//        return name;
//    };
//
//};
//
//
//class inputdata{
//    // TODO: not all public. Can we circumvent this class to have cleaner code?
//    // static json all_data;
//    unordered_map<string, trace*> tail_map;
//    list<trace*> all_traces;
//
//    static vector<string> alphabet;
//    static map<string, int> r_alphabet;
//
//    static vector<string> types;
//    static map<string, int> r_types;
//
//    static vector<attribute> trace_attributes;
//    static vector<attribute> symbol_attributes;
//
//    int max_sequences;
//    int num_sequences;
//    int node_number;
//    int num_tails;
//
//    /* for reading csv files */
//    set<int> id_cols;
//    set<int> type_cols;
//    set<int> symbol_cols;
//    set<int> data_cols;
//    set<int> trace_attr_cols;
//    set<int> symbol_attr_cols;
//
//public:
//
//    /* input reading, we read from abbadingo format */
//    void read_json_file(istream &input_stream);
//    void read_abbadingo_file(istream &input_stream);
//    void read_csv_file(istream &input_stream);
//    trace* read_csv_row(istream &input_stream);
//
//    void read_abbadingo_sequence(istream &input_stream, trace*);
//    void read_abbadingo_type(istream &input_stream, trace*);
//    void read_abbadingo_symbol(istream &input_stream, tail*);
//    void add_trace_to_apta(trace *t, apta *the_apta);
//
//    list<trace*>::iterator traces_start(){ return all_traces.begin(); }
//    list<trace*>::iterator traces_end(){ return all_traces.end(); }
//
//    static inline string& get_symbol(int a){
//        return alphabet[a];
//    }
//    static inline int get_reverse_symbol(string a){
//        return r_alphabet[a];
//    }
//    static inline string& get_type(int a){
//        return types[a];
//    }
//    static inline int get_reverse_type(string a){
//        return r_types[a];
//    }
//    /* gets an attribute, first symbol attributes, then trace attributes */
//    static inline attribute* get_trace_attribute(int attr){
//        if(attr < trace_attributes.size()){
//            return &trace_attributes[attr];
//        }
//        return nullptr;
//    }
//    static inline attribute* get_symbol_attribute(int attr){
//        if(attr < symbol_attributes.size()){
//            return &symbol_attributes[attr];
//        }
//        return nullptr;
//    }
//    static inline attribute* get_attribute(int attr){
//        if(attr < symbol_attributes.size()){
//            return &symbol_attributes[attr];
//        }
//        attr = attr - symbol_attributes.size();
//        if(attr < trace_attributes.size()){
//            return &trace_attributes[attr];
//        }
//        return nullptr;
//    }
//    static inline int get_num_symbol_attributes() {
//        return symbol_attributes.size();
//    }
//    static inline int get_num_trace_attributes() {
//        return trace_attributes.size();
//    }
//    static inline int get_num_attributes() {
//        return get_num_trace_attributes() + get_num_symbol_attributes();
//    }
//
//    /* attribute properties:
//     * splittable: will be used to infer guards
//     * distributionable: will be used in evaluation functions that model attributes
//     * discrete: whether the attribute discrete or continuous
//     * target: will be used in evaluation functions as class/target/prediction variable
//     * */
//    static inline bool is_splittable(int attr){
//        return get_attribute(attr)->splittable;
//    }
//    static inline bool is_distributionable(int attr) {
//        return get_attribute(attr)->distributionable;
//    }
//    static inline bool is_discrete(int attr) {
//        return get_attribute(attr)->discrete;
//    }
//    static inline bool is_target(int attr) {
//        return get_attribute(attr)->target;
//    }
//
//    /* inputdata properties:
//     * the number of distinct sequence types
//     * the size of the input data
//     * */
//    static inline int get_types_size(){
//        return types.size();
//    }
//    static inline int get_alphabet_size(){
//        return alphabet.size();
//    }
//
//    // to init counters etc
//    inputdata();
//    // to delete input traces
//    ~inputdata();
//
//    static inline int symbol_from_string(string symbol){
//        if(r_alphabet.find(symbol) == r_alphabet.end()){
//            r_alphabet[symbol] = alphabet.size();
//            alphabet.push_back(symbol);
//        }
//        return r_alphabet[symbol];
//    }
//    static inline string string_from_symbol(int symbol) {
//        if(symbol == -1) return "fin";
//        if(alphabet.size() < symbol) return "_";
//        return alphabet[symbol];
//    }
//    static inline int type_from_string(string type){
//        if(r_types.find(type) == r_types.end()){
//            r_types[type] = types.size();
//            types.push_back(type);
//        }
//        return r_types[type];
//    }
//    static inline string string_from_type(int type) {
//        return types[type];
//    }
//
//    inline int get_num_nodes() {
//        return node_number;
//    }
//    inline int get_num_sequences() {
//        return num_sequences;
//    }
//    inline int get_max_sequences() {
//        return max_sequences;
//    }
//
//    void read_abbadingo_header(istream &input_stream);
//    void read_csv_header(istream &input_stream);
//
//    static trace *access_trace(tail *t);
//    static tail *access_tail(tail *t);
//
//    friend class tail;
//    friend class trace;
//
//    void add_traces_to_apta(apta *the_apta);
//};
//
//class trace{
//public:
//    int sequence;
//    int length;
//    int type;
//    double* trace_attr;
//
//    int refs;
//
//    tail* head;
//    tail* end_tail;
//
//    trace();
//    ~trace();
//    void initialize();
//
//    inline int get_type() const{ return type; }
//    inline int get_length(){ return length; }
//    inline int get_sequence(){ return sequence; }
//    inline tail* get_head(){ return head; }
//    inline tail* get_end(){ return end_tail; }
//    inline void inc_refs(){ ++refs; }
//    void erase();
//
//    string to_string();
//
//    void reverse();
//};
//
//class tail_data{
//public:
//    int index;
//    int symbol;
//    double* attr;
//
//    string data;
//
//    int tail_nr;
//
//    tail_data();
//    ~tail_data();
//    void initialize();
//};
//
//class tail{
//public:
//    tail();
//    tail(tail *ot);
//    ~tail();
//    void initialize(tail* ot);
//
//    tail_data* td;
//    trace* tr;
//
//    tail* future_tail;
//    tail* past_tail;
//    tail* next_in_list;
//    tail* split_from;
//    tail* split_to;
//
//
//
//    void split(tail* t);
//    void undo_split();
//    tail* next() const;
//    tail* future() const;
//    tail* past() const;
//    tail* split_to_end();
//
//    inline int get_index(){
//        return td->index;
//    }
//    inline int get_type(){
//        return tr->type;
//    }
//    inline int get_length(){
//        return tr->length;
//    }
//    inline int get_sequence(){
//        return tr->sequence;
//    }
//    inline int get_symbol(){
//        return td->symbol;
//    }
//    inline double get_symbol_value(int attr){
//        return td->attr[attr];
//    }
//    inline double get_trace_value(int attr){
//        return tr->trace_attr[attr];
//    }
//    inline double get_value(int attr){
//        int num_trace_attributes = inputdata::trace_attributes.size();
//        if(attr < num_trace_attributes)
//            return tr->trace_attr[attr];
//        return td->attr[attr - num_trace_attributes];
//    }
//    inline string get_data(){
//        return td->data;
//    }
//    inline bool is_final(){
//        return td->symbol == -1;
//    }
//
//    inline int get_nr(){
//        return td->tail_nr;
//    }
//
//    void set_future(tail* ft);
//    string to_string();
//};


#endif /* _INPUTDATA_H_*/
