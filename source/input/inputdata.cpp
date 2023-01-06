#include "input/inputdata.h"
#include "apta.h"

using namespace std;

string &inputdata::get_symbol(int a) {
    return alphabet[a];
}

int inputdata::get_reverse_symbol(string a) {
    return r_alphabet[a];
}

std::string &inputdata::get_type(int a) {
    return types[a];
}

int inputdata::get_reverse_type(std::string a) {
    return r_types[a];
}

attribute *inputdata::get_trace_attribute(int attr) {
    if(attr < trace_attributes.size()){
        return &trace_attributes[attr];
    }
    return nullptr;
}

attribute *inputdata::get_symbol_attribute(int attr) {
    if(attr < symbol_attributes.size()){
        return &symbol_attributes[attr];
    }
    return nullptr;
}

attribute *inputdata::get_attribute(int attr) {
    if(attr < symbol_attributes.size()){
        return &symbol_attributes[attr];
    }
    attr = attr - symbol_attributes.size();
    if(attr < trace_attributes.size()){
        return &trace_attributes[attr];
    }
    return nullptr;
}

int inputdata::get_num_symbol_attributes() {
    return symbol_attributes.size();
}

int inputdata::get_num_trace_attributes() {
    return trace_attributes.size();
}

int inputdata::get_num_attributes() {
    return get_num_trace_attributes() + get_num_symbol_attributes();
}

bool inputdata::is_splittable(int attr) {
    return get_attribute(attr)->splittable;
}

bool inputdata::is_distributionable(int attr) {
    return get_attribute(attr)->distributionable;
}

bool inputdata::is_discrete(int attr) {
    return get_attribute(attr)->discrete;
}

bool inputdata::is_target(int attr) {
    return get_attribute(attr)->target;
}

int inputdata::get_types_size() {
    return types.size();
}

int inputdata::get_alphabet_size() {
    return alphabet.size();
}

int inputdata::symbol_from_string(std::string symbol) {
    if(r_alphabet.find(symbol) == r_alphabet.end()){
        r_alphabet[symbol] = alphabet.size();
        alphabet.push_back(symbol);
    }
    return r_alphabet[symbol];
}

std::string inputdata::string_from_symbol(int symbol) {
    if(symbol == -1) return "fin";
    if(alphabet.size() < symbol) return "_";
    return alphabet[symbol];
}

int inputdata::type_from_string(std::string type) {
    if(r_types.find(type) == r_types.end()){
        r_types[type] = types.size();
        types.push_back(type);
    }
    return r_types[type];
}

std::string inputdata::string_from_type(int type) {
    return types[type];
}

void inputdata::add_traces_to_apta(apta *the_apta) {
    for(auto* tr : traces){
        add_trace_to_apta(tr, the_apta);
        if(!ADD_TAILS) tr->erase();
    }
}

void inputdata::add_trace_to_apta(trace *tr, apta *the_apta) {
    int depth = 0;
    apta_node* node = the_apta->root;
    /*if(node->access_trace == nullptr){
        node->access_trace = mem_store::create_trace();
    }*/

    if(REVERSE_TRACES){
        tr->reverse();
    }

    tail* t = tr->head;

    while(t != nullptr){
        node->size = node->size + 1;
        if(ADD_TAILS) node->add_tail(t);
        node->data->add_tail(t);

        depth++;
        if(t->is_final()){
            node->final = node->final + 1;
        } else {
            int symbol = t->get_symbol();
            if(node->child(symbol) == nullptr){
                if(node->size < PARENT_SIZE_THRESHOLD){
                    break;
                }
                auto* next_node = mem_store::create_node(nullptr);
                node->set_child(symbol, next_node);
                next_node->source = node;
                //next_node->access_trace = inputdata::access_trace(t);
                next_node->depth  = depth;
                next_node->number = ++(this->node_number);
            }
            node = node->child(symbol)->find();
        }
        t = t->future();
    }
}

trace *inputdata::access_trace(tail *t) {
    t = t->split_to_end();
    int length = 1;
    trace* tr = mem_store::create_trace(this);
    tr->sequence = t->tr->sequence;
    tr->type = t->tr->type;
    for(int i = 0; i < this->get_num_trace_attributes(); ++i){
        tr->trace_attr[i] = t->tr->trace_attr[i];
    }
    if(STORE_ACCESS_STRINGS){
        tail* ti = t->tr->head->split_to_end();
        tail* tir = this->access_tail(ti);
        tr->head = tir;
        tir->tr = tr;
        tail* temp = tr->head;
        while(ti != t){
            length++;
            ti = ti->future();
            temp = this->access_tail(ti);
            tir->set_future(temp);
            tir = temp;
            tir->tr = tr;
        }
        tr->refs = 1;
        tr->length = length;
        tr->end_tail = temp;
    } else {
        tr->head = this->access_tail(t);
        tr->refs = 1;
        tr->length = 1;
        tr->end_tail = tr->head;
    }
    return tr;
}

tail *inputdata::access_tail(tail *t) {
    tail* res = mem_store::create_tail(nullptr);
    res->td->index = t->td->index;
    res->td->symbol = t->td->symbol;
    for(int i = 0; i < this->get_num_symbol_attributes(); ++i){
        res->td->attr[i] = t->td->attr[i];
    }
    res->td->data = t->td->data;
    return res;
}

int inputdata::get_num_sequences() {
    return num_sequences;
}

int inputdata::get_max_sequences() {
    return max_sequences;
}
