#include "i_inputdata.h"
#include "apta.h"

using namespace std;

string &IInputData::get_symbol(int a) {
    return alphabet[a];
}

int IInputData::get_reverse_symbol(string a) {
    return r_alphabet[a];
}

std::string &IInputData::get_type(int a) {
    return types[a];
}

int IInputData::get_reverse_type(std::string a) {
    return r_types[a];
}

Attribute *IInputData::get_trace_attribute(int attr) {
    if(attr < trace_attributes.size()){
        return &trace_attributes[attr];
    }
    return nullptr;
}

Attribute *IInputData::get_symbol_attribute(int attr) {
    if(attr < symbol_attributes.size()){
        return &symbol_attributes[attr];
    }
    return nullptr;
}

Attribute *IInputData::get_attribute(int attr) {
    if(attr < symbol_attributes.size()){
        return &symbol_attributes[attr];
    }
    attr = attr - symbol_attributes.size();
    if(attr < trace_attributes.size()){
        return &trace_attributes[attr];
    }
    return nullptr;
}

int IInputData::get_num_symbol_attributes() {
    return symbol_attributes.size();
}

int IInputData::get_num_trace_attributes() {
    return trace_attributes.size();
}

int IInputData::get_num_attributes() {
    return get_num_trace_attributes() + get_num_symbol_attributes();
}

bool IInputData::is_splittable(int attr) {
    return get_attribute(attr)->splittable;
}

bool IInputData::is_distributionable(int attr) {
    return get_attribute(attr)->distributionable;
}

bool IInputData::is_discrete(int attr) {
    return get_attribute(attr)->discrete;
}

bool IInputData::is_target(int attr) {
    return get_attribute(attr)->target;
}

int IInputData::get_types_size() {
    return types.size();
}

int IInputData::get_alphabet_size() {
    return alphabet.size();
}

int IInputData::symbol_from_string(std::string symbol) {
    if(r_alphabet.find(symbol) == r_alphabet.end()){
        r_alphabet[symbol] = alphabet.size();
        alphabet.push_back(symbol);
    }
    return r_alphabet[symbol];
}

std::string IInputData::string_from_symbol(int symbol) {
    if(symbol == -1) return "fin";
    if(alphabet.size() < symbol) return "_";
    return alphabet[symbol];
}

int IInputData::type_from_string(std::string type) {
    if(r_types.find(type) == r_types.end()){
        r_types[type] = types.size();
        types.push_back(type);
    }
    return r_types[type];
}

std::string IInputData::string_from_type(int type) {
    return types[type];
}

void IInputData::add_traces_to_apta(apta *the_apta) {
    for(auto* tr : traces){
        add_trace_to_apta(tr, the_apta);
        if(!ADD_TAILS) tr->erase();
    }
}

void IInputData::add_trace_to_apta(Trace *tr, apta *the_apta) {
    int depth = 0;
    apta_node* node = the_apta->root;
    /*if(node->access_trace == nullptr){
        node->access_trace = mem_store::create_trace();
    }*/

    if(REVERSE_TRACES){
        tr->reverse();
    }

    Tail* t = tr->head;

    // TODO: re enable after rename
//    while(t != nullptr){
//        node->size = node->size + 1;
//        if(ADD_TAILS) node->add_tail(t);
//        node->data->add_tail(t);
//
//        depth++;
//        if(t->is_final()){
//            node->final = node->final + 1;
//        } else {
//            int symbol = t->get_symbol();
//            if(node->child(symbol) == nullptr){
//                if(node->size < PARENT_SIZE_THRESHOLD){
//                    break;
//                }
//                auto* next_node = mem_store::create_node(nullptr);
//                node->set_child(symbol, next_node);
//                next_node->source = node;
//                //next_node->access_trace = inputdata::access_trace(t);
//                next_node->depth  = depth;
//                next_node->number = ++(this->node_number);
//            }
//            node = node->child(symbol)->find();
//        }
//        t = t->future();
//    }
}
