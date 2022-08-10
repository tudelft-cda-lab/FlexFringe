#ifndef FLEXFRINGE_TAIL_H
#define FLEXFRINGE_TAIL_H

#include <string>
#include "input/trace.h"
#include "trace.h"

class Trace;

class TailData{
public:
    int index;
    int symbol;
    double* attr;

    std::string data;

    int tail_nr;

    TailData();
    ~TailData();
    void initialize();
};

class Tail{
public:
    Tail();
    Tail(Tail *ot);
    ~Tail();
    void initialize(Tail* ot);

    TailData* td;
    Trace* tr;

    Tail* future_tail;
    Tail* past_tail;
    Tail* next_in_list;
    Tail* split_from;
    Tail* split_to;

    void split(Tail* t);
    void undo_split();
    Tail* next() const;
    Tail* future() const;
    Tail* past() const;
    Tail* split_to_end();

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

    void set_future(Tail* ft);
    std::string to_string();
};

#endif //FLEXFRINGE_TAIL_H
