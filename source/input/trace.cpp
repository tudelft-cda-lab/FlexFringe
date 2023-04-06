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

/**
 * @brief This function helps us debugging.
 * 
 * @param t The type.
 * @return std::string The mapped type. 
 */
string trace::get_mapped_type(int t) const {
    return this->inputData->string_from_type(t);
} 

/**
 * Adds a sentinel tail (with symbol -1) at the end of this trace
 */
void trace::finalize() {
    tail* new_end_tail = mem_store::create_tail(nullptr);
    new_end_tail->tr = this;
    new_end_tail->td->index = this->end_tail->get_index() + 1;

    // Add final tail to trace
    this->end_tail->set_future(new_end_tail);
    this->end_tail = new_end_tail;
}

/**
 * Pops the front tail of a trace
 */
void trace::pop_front() {
    tail* old_head = this->head;
    if (old_head == nullptr) { return; }

    this->head = old_head->future();
    this->length--;

    // TODO: should we delete? may cause issues with tail data being reused
    // Lets just not delete for now, and technically "leak" the tail memory
    // since the tail data may be used elsewhere still?
    //mem_store::delete_tail(old_head);

    if (this->head == nullptr) { return; }
    this->head->past_tail = nullptr;
}

/**
 * @brief Gets the input sequence from the trace. Differentiates access-traces
 * and non access-traces. 
 * 
 * Access traces are not finalized, hence need to be treated differently, or 
 * else the while-loop would ignore the last element.
 * 
 * @param is_access_trace Boolean indicating whether it is an access trace or not. Default = false.
 * @return const std::vector<int> The sequence of the trace.
 */
const std::vector<int> trace::get_input_sequence(const bool is_access_trace) const {
    std::vector<int> res;
    tail* t = head;
    if(t->get_symbol() == -1) return std::vector<int>(); // empty strings

    while((!is_access_trace && t != end_tail) || (is_access_trace && t != nullptr)){
        res.push_back(t->get_symbol());
        t = t->future();
    }
    return res;
}