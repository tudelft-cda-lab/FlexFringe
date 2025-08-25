#include "input/trace.h"
#include "input/inputdata.h"
#include "mem_store.h"
#include "input/tail.h"

trace::trace(inputdata* inputData) {
    initialize(inputData);
}

/**
 * @brief Construct a new trace::trace object
 * 
 * Do copies of the tails leading to this one as well. We do
 * NOT copy the final element indicating that the end of the trace 
 * if there is one.
 * 
 * @param id Inputdata
 * @param other The other strace
 */
trace::trace(inputdata* id, trace* other){
    initialize(id, other);

    head = mem_store::create_tail(other->head);
    tail* other_tail = other->head;
    tail* this_tail = this->head;
    this_tail->tr = this;

    while(other_tail->get_symbol() != -1){
        tail* new_tail = mem_store::create_tail(other_tail);
        this_tail->future_tail = new_tail;
        new_tail->past_tail = this_tail;
        new_tail->tr = this;

        this_tail = new_tail;
        other_tail = other_tail->future_tail;
    }

    end_tail = this_tail;
}

void trace::initialize(inputdata* inputData, trace* other_trace) {
    this->inputData = inputData;
    if(other_trace == nullptr){    
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
    else{
        length = other_trace->length;
        type = other_trace->type;
        sequence = other_trace->sequence;
        trace_attr = new double[inputData->get_num_trace_attributes()];
        for(int i = 0; i < this->inputData->get_num_trace_attributes(); ++i){
            trace_attr[i] = other_trace->trace_attr[i];
        }
        refs = other_trace->refs;

        // copy tail content of other trace
        head = mem_store::create_tail(other_trace->head);
        tail* other_tail = other_trace->head;
        tail* this_tail = this->head;
        this_tail->tr = this;

        while(other_tail->get_symbol() != -1){
            tail* new_tail = mem_store::create_tail(other_tail);
            this_tail->future_tail = new_tail;
            new_tail->past_tail = this_tail;
            new_tail->tr = this;

            this_tail = new_tail;
            other_tail = other_tail->future_tail;
        }

        end_tail = this_tail;    
    }
}

trace::~trace(){
    delete trace_attr;
    delete head;
}

#if defined(_MSC_VER) 
void trace::erase(){
#else
void __attribute__((optimize("O0"))) trace::erase(){
#endif
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

std::string trace::to_string(){
    if(this == nullptr) return "NULL";
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
std::string trace::get_mapped_type(int t) const {
    return this->inputData->string_from_type(t);
} 

/**
 * Adds a sentinel tail (with symbol -1) at the end of this trace
 */
void trace::finalize() {
    if (!this->is_finalized()) {
        tail* new_end_tail = mem_store::create_tail(nullptr);
        new_end_tail->tr = this;
        new_end_tail->td->index = this->end_tail->get_index() + 1;

        // Add final tail to trace
        this->end_tail->set_future(new_end_tail);
        this->end_tail = new_end_tail;
    }
}

/**
 * A trace is finalized when the last tail has a symbol of -1
 * which is not counted against the lenght of the trace
 * @return
 */
bool trace::is_finalized() {
    return this->end_tail->is_final();
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
 * @param prepare_empty_slot Optimization. If true, the vector will have size of access trace plus one. 
 * Helps us avoiding additional resizing in L# algorithm (minor optimization). Default = false.
 * @return const std::vector<int> The sequence of the trace.
 */
const std::vector<int> trace::get_input_sequence(const bool is_access_trace, const bool prepare_empty_slot) const {
    tail* t = head;
    if(t == nullptr || t->get_symbol() == -1) return prepare_empty_slot ? std::vector<int>(1) : std::vector<int>(); // empty strings

    std::vector<int> res = prepare_empty_slot ? std::vector<int>(this->length + 1) : std::vector<int>(this->length);
    int idx = 0;
    while((!is_access_trace && t != end_tail) || (is_access_trace && t != nullptr)){
        res[idx] = t->get_symbol();
        ++idx;
        t = t->future();
    }
    return res;
}
