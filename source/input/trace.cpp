#include "input/trace.h"
#include "input/inputdata.h"
#include "mem_store.h"
#include "input/tail.h"

trace::trace(inputdata* inputData) {
    initialize(inputData);
}

void trace::initialize(inputdata* inputData) {
    this->inputData = inputData;
    sequence = -1;
    length = -1;
    type = -1;
    trace_attr = new double[inputData->get_num_trace_attributes()];
    for(int i = 0; i < this->inputData->get_num_trace_attributes(); ++i){
        trace_attr[i] = 0.0;
    }
    head = nullptr;
    end_tail = nullptr;
    refs = 1;
}

trace::~trace(){
    delete trace_attr;
    delete head;
}

void trace::erase(){
    --refs;
    if(refs == 0) mem_store::delete_trace(this);
}

void trace::reverse(){
    tail* t = head;
    while(t != end_tail){
        tail* temp_future = t->future_tail;
        t->future_tail = t->past_tail;
        t->past_tail = temp_future;
        t = temp_future;
    }
    tail* temp_head = head;
    head = end_tail->past_tail;
    head->past_tail = nullptr;
    end_tail->past_tail = temp_head;
    temp_head->future_tail = end_tail;
}

string trace::to_string(){
    tail* t = head;
    if(t == nullptr) return "0 0";
    while(t->future() != nullptr) t = t->future();
    return "" + this->inputData->string_from_type(get_type()) + " " + std::to_string(get_length()) + " " + t->to_string() +"";
}