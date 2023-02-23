#ifndef FLEXFRINGE_TRACE_H
#define FLEXFRINGE_TRACE_H

//#import "inputdata.h"
#include <string>

class inputdata;
class tail;

class trace {
private:
    friend class tail;
    friend class apta_node;

    inputdata* inputData;

    // Only mem store is allowed to create and destroy
    friend class mem_store;
    explicit trace(inputdata*);
    ~trace();

public:

    trace() = delete;

    int sequence;
    int length;
    int type;
    double* trace_attr;

    int refs;

    tail* head;
    tail* end_tail;

    void initialize(inputdata *inputData);
    void finalize();
    bool is_finalized();

    inline int get_type() const{ return type; }
    inline int get_length(){ return length; }
    inline int get_sequence(){ return sequence; }
    inline tail* get_head(){ return head; }
    inline tail* get_end(){ return end_tail; }
    inline void inc_refs(){ ++refs; }
    void erase();

    std::string to_string();

    void reverse();
    void pop_front();
};


#endif //FLEXFRINGE_TRACE_H
