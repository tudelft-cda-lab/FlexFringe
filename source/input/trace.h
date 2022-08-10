#ifndef FLEXFRINGE_TRACE_H
#define FLEXFRINGE_TRACE_H

//#import "inputdata.h"
#include <string>

class IInputData;
class Tail;

class Trace {
private:
    friend class Tail;
    IInputData* inputData;

    // Only mem store is allowed to create and destroy
    friend class mem_store;
    explicit Trace(IInputData*);
    ~Trace();

public:

    Trace() = delete;

    int sequence;
    int length;
    int type;
    double* trace_attr;

    int refs;

    Tail* head;
    Tail* end_tail;

    void initialize(IInputData *inputData);

    inline int get_type() const{ return type; }
    inline int get_length(){ return length; }
    inline int get_sequence(){ return sequence; }
    inline Tail* get_head(){ return head; }
    inline Tail* get_end(){ return end_tail; }
    inline void inc_refs(){ ++refs; }
    void erase();

    std::string to_string();

    void reverse();
};


#endif //FLEXFRINGE_TRACE_H
