#include <sstream>

#include "input/tail.h"
#include "input/trace.h"
#include "input/i_inputdata.h"
#include "input/inputdatalocator.h"

using namespace std;

/* Tail list functions, they are in two lists:
 * 1) past_tail <-> current_tail <-> future_tail
 * 2) split_from <-> current_tail <-> split_to
 * 1 is used to get past and futures from the input data sequences
 * 2 is used to keep track of and be able to quickly undo_merge split refinements
 * */
void Tail::split(Tail* t){
    t->split_from = this;
    t->future_tail = future_tail;
    t->past_tail = past_tail;
    split_to = t;
}

void Tail::undo_split(){
    split_to = nullptr;
}

Tail* Tail::next() const{
    if(split_to == nullptr) return next_in_list;
    if(next_in_list == nullptr) return nullptr;
    return next_in_list->next();
}

Tail* Tail::split_to_end(){
    if(split_to == nullptr) return this;
    return split_to->split_to_end();
}

Tail* Tail::future() const{
    if(future_tail == nullptr) return nullptr;
    if(split_to == nullptr) return future_tail->split_to_end();
    return split_to->future();
}

Tail* Tail::past() const{
    if(past_tail == nullptr) return nullptr;
    if(split_to == nullptr) return past_tail->split_to_end();
    return split_to->past();
}

void Tail::set_future(Tail* ft){
    future_tail = ft;
    ft->past_tail = this;
}

/* tail constructors
 * copies values from existing tail but does not put into any list
 * initialize_after_adding_traces re-used a previously declared but erased tail */
Tail::Tail(){
    td = new TailData();
    past_tail = nullptr;
    future_tail = nullptr;
    next_in_list = nullptr;
    split_from = nullptr;
    split_to = nullptr;

    tr = nullptr;
}

Tail::Tail(Tail* ot){
    if(ot != nullptr){
        td = ot->td;
        tr = ot->tr;
    } else {
        td = new TailData();
        tr = nullptr;
    }
    past_tail = nullptr;
    future_tail = nullptr;
    next_in_list = nullptr;
    split_from = nullptr;
    split_to = nullptr;
}

void Tail::initialize(Tail* ot){
    if(ot != nullptr){
        td = ot->td;
        tr = ot->tr;
    } else {
        td = new TailData();
        tr = nullptr;
    }
    past_tail = nullptr;
    future_tail = nullptr;
    next_in_list = nullptr;
    split_from = nullptr;
    split_to = nullptr;
}


string Tail::to_string(){
    auto inputdata = this->tr->inputData;

    ostringstream ostr;
    Tail* t = this;
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

Tail::~Tail(){
    if(split_from == nullptr){
        delete td;
    }
    //if(future_tail != nullptr) delete future_tail;
}



int Tail::get_index(){
    return td->index;
}
int Tail::get_type(){
    return tr->type;
}
int Tail::get_length(){
    return tr->length;
}
int Tail::get_sequence(){
    return tr->sequence;
}
int Tail::get_symbol(){
    return td->symbol;
}
double Tail::get_symbol_value(int attr){
    return td->attr[attr];
}
double Tail::get_trace_value(int attr){
    return tr->trace_attr[attr];
}
double Tail::get_value(int attr){
    int num_trace_attributes = this->tr->inputData->get_num_trace_attributes();
    if(attr < num_trace_attributes)
        return tr->trace_attr[attr];
    return td->attr[attr - num_trace_attributes];
}
std::string Tail::get_data(){
    return td->data;
}
bool Tail::is_final(){
    return td->symbol == -1;
}

int Tail::get_nr(){
    return td->tail_nr;
}

// TODO
TailData::TailData() {
    auto inputdata = InputDataLocator::get();
    index = -1;
    symbol = -1;
    attr = new double[inputdata->get_num_symbol_attributes()];
    for(int i = 0; i < inputdata->get_num_symbol_attributes(); ++i){
        attr[i] = 0.0;
    }
    data = "";
    tail_nr = -1;
}

// TODO
TailData::~TailData() {

}

// TODO
void TailData::initialize() {
    auto inputdata = InputDataLocator::get();
    index = -1;
    symbol = -1;
    attr = new double[inputdata->get_num_symbol_attributes()];
    for(int i = 0; i < inputdata->get_num_symbol_attributes(); ++i){
        attr[i] = 0.0;
    }
    data = "";
    tail_nr = -1;
}
