#include <sstream>

#include "input/tail.h"
#include "input/trace.h"
#include "input/inputdata.h"
#include "input/inputdatalocator.h"

using namespace std;

/* tail list functions, they are in two lists:
 * 1) past_tail <-> current_tail <-> future_tail
 * 2) split_from <-> current_tail <-> split_to
 * 1 is used to get past and futures from the input data sequences
 * 2 is used to keep track of and be able to quickly undo_merge split refinements
 * */
void tail::split(tail* t){
    t->split_from = this;
    t->future_tail = future_tail;
    t->past_tail = past_tail;
    split_to = t;
}

void tail::undo_split(){
    split_to = nullptr;
}

tail* tail::next() const{
    if(split_to == nullptr) return next_in_list;
    if(next_in_list == nullptr) return nullptr;
    return next_in_list->next();
}

tail* tail::split_to_end(){
    if(split_to == nullptr) return this;
    return split_to->split_to_end();
}

tail* tail::future() const{
    if(future_tail == nullptr) return nullptr;
    if(split_to == nullptr) return future_tail->split_to_end();
    return split_to->future();
}

tail* tail::past() const{
    if(past_tail == nullptr) return nullptr;
    if(split_to == nullptr) return past_tail->split_to_end();
    return split_to->past();
}

void tail::set_future(tail* ft){
    future_tail = ft;
    ft->past_tail = this;
}

/* tail constructors
 * copies values from existing tail but does not put into any list
 * initialize_after_adding_traces re-used a previously declared but erased tail */
tail::tail(){
    td = std::make_shared<tail_data>();
    past_tail = nullptr;
    future_tail = nullptr;
    next_in_list = nullptr;
    split_from = nullptr;
    split_to = nullptr;

    tr = nullptr;
}

tail::tail(tail* ot){
    if(ot != nullptr){
        td = ot->td;
        tr = ot->tr;
    } else {
        td = std::make_shared<tail_data>();
        tr = nullptr;
    }
    past_tail = nullptr;
    future_tail = nullptr;
    next_in_list = nullptr;
    split_from = nullptr;
    split_to = nullptr;
}

void tail::initialize(tail* ot){
    if(ot != nullptr){
        td = ot->td;
        tr = ot->tr;
    } else {
        td = std::make_shared<tail_data>();
        tr = nullptr;
    }
    past_tail = nullptr;
    future_tail = nullptr;
    next_in_list = nullptr;
    split_from = nullptr;
    split_to = nullptr;
}


string tail::to_string(){
    auto inputdata = this->tr->inputData;

    ostringstream ostr;
    tail* t = this;
    while(t->past() != nullptr) t = t->past();

    while(t != this->future() && !t->is_final()){
        ostr << inputdata->get_symbol(t->get_symbol());
        if(inputdata->get_num_symbol_attributes() > 0){
            ostr << ":";
            for(int i = 0; i < inputdata->get_num_symbol_attributes(); i++){
                ostr << t->get_symbol_value(i);
                if(i + 1 < inputdata->get_num_symbol_attributes())
                    ostr << ",";
            }
        }
        if(t->get_data() != "") {
            ostr << "/" << t->get_data();
        }
        t = t->future_tail;
        if(t != this->future_tail && !t->is_final()){
            ostr << " ";
        }
    }
    return ostr.str();
}

tail::~tail(){
    if(split_from == nullptr){
        // Set shared taildata pointer to null
        td = nullptr;
    }
    //if(future_tail != nullptr) delete future_tail;
}



int tail::get_index(){
    return td->index;
}
int tail::get_type(){
    return tr->type;
}
int tail::get_length(){
    return tr->length;
}
int tail::get_sequence(){
    return tr->sequence;
}
int tail::get_symbol(){
    return td->symbol;
}
double tail::get_symbol_value(int attr){
    return td->attr[attr];
}
double tail::get_trace_value(int attr){
    return tr->trace_attr[attr];
}
double tail::get_value(int attr){
    int num_trace_attributes = this->tr->inputData->get_num_trace_attributes();
    if(attr < num_trace_attributes)
        return tr->trace_attr[attr];
    return td->attr[attr - num_trace_attributes];
}
std::string tail::get_data(){
    return td->data;
}
bool tail::is_final(){
    return td->symbol == -1;
}

int tail::get_nr(){
    return td->tail_nr;
}

tail_data::tail_data() {
    auto inputdata = inputdata_locator::get();
    index = -1;
    symbol = -1;
    attr = std::make_unique<double[]>(inputdata->get_num_symbol_attributes());
    for(int i = 0; i < inputdata->get_num_symbol_attributes(); ++i){
        attr[i] = 0.0;
    }
    data = "";
    tail_nr = -1;
}


tail_data::~tail_data() = default;

void tail_data::initialize() {
    auto inputdata = inputdata_locator::get();
    index = -1;
    symbol = -1;
    attr = std::make_unique<double[]>(inputdata->get_num_symbol_attributes());
    for(int i = 0; i < inputdata->get_num_symbol_attributes(); ++i){
        attr[i] = 0.0;
    }
    data = "";
    tail_nr = -1;
}
