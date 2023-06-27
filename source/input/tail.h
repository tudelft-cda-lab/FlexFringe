#ifndef FLEXFRINGE_TAIL_H
#define FLEXFRINGE_TAIL_H

#include <string>
#include "input/trace.h"
#include "trace.h"

class trace;

class tail_data{
public:
    int index;
    int symbol;
    double* attr;

    // statistics only used with active learning
    int statistics; 
    int final_statistics;

    std::string data;

    int tail_nr;

    tail_data();
    ~tail_data();
    void initialize();
};

class tail{
public:
    tail();
    tail(tail *ot);
    ~tail();
    void initialize(tail* ot);

    tail_data* td;
    trace* tr;

    tail* future_tail;
    tail* past_tail;
    tail* next_in_list;
    tail* split_from;
    tail* split_to;

    void split(tail* t);
    void undo_split();
    tail* next() const;
    tail* future() const;
    tail* past() const;
    tail* split_to_end();

    int get_index();
    int get_type();
    int get_length();
    int get_sequence();
    int get_symbol();
    double get_symbol_value(int attr);
    double get_trace_value(int attr);
    double get_value(int attr);
    std::string get_data();
    bool is_final();
    int get_nr();

    void update_statistics(int s){ this->td->statistics = s; }
    int get_statistics(){ return this->td->statistics; }
    void update_final_statistics(int fs){ this->td->final_statistics = fs; }
    int get_final_statistics(){ return this->td->final_statistics; }

    void set_future(tail* ft);
    std::string to_string();
};

#endif //FLEXFRINGE_TAIL_H
