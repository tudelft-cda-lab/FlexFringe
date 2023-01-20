#include "input/inputdata.h"
#include "apta.h"
#include "stringutil.h"

using namespace std;

/**
 * Read symbols from a parser and construct inputdata from them.
 * @param input_parser the input parser to read from
 */
void inputdata::read(parser* input_parser) {
    std::unordered_map<std::string, trace*> trace_map;

    while(true) {
        // Do we have another symbol from the parser?
        auto cur_symbol_maybe = input_parser->next();
        if (!cur_symbol_maybe.has_value()) {
            break;
        }

        // Process it!
        auto cur_symbol = cur_symbol_maybe.value();
        process_symbol_info(cur_symbol, trace_map);
    }

    // Add all collected traces to the list of traces
    for (const auto &[id, trace]: trace_map) {
        traces.push_back(trace);
    }

    // Make sure they are ordered by sequence number
    traces.sort([](auto& left, auto& right) {
        return left->sequence < right->sequence;
    });
}

/**
 * Reads the input symbols from the given parser, but creates sliding windows over the input.
 * e.g. a trace [a, b, c, d] with sliding window size 3 turns into two actual traces in the inputdata:
 * [a, b, c] and [b, c, d]
 *
 * @param input_parser input parser to read symbols from
 * @param sliding_window_size sliding window size
 * @param sliding_window_stride sliding window stride
 * @param sliding_window_type add types to the sliding window traces
 */
void inputdata::read_slidingwindow(parser* input_parser,
                                   ssize_t sliding_window_size,
                                   ssize_t sliding_window_stride,
                                   bool sliding_window_type) {

    std::unordered_map<std::string, trace*> trace_map;

    while(true) {
        auto cur_symbol_maybe = input_parser->next();
        if (!cur_symbol_maybe.has_value()) {
            break;
        }

        auto cur_symbol = cur_symbol_maybe.value();

        auto [tr, new_tail] = process_symbol_info(cur_symbol, trace_map);

        // Sliding window stuff.
        // Definitely some more refactoring potential in here
        if(tr->get_length() == sliding_window_size){
            if(sliding_window_type){
                string type_string = inputdata::string_from_symbol(new_tail->get_symbol());
                if(r_types.find(type_string) == r_types.end()){
                    r_types[type_string] = (int)types.size();
                    types.push_back(type_string);
                }
                tr->type = r_types[type_string];
            }
            trace* new_window = mem_store::create_trace();
            new_window->type = tr->type;
            new_window->sequence = inputdata::num_sequences;
            tail* t = tr->get_head();
            int index = 0;
            tail* new_window_tail = nullptr;
            while(t != nullptr){
                if(index >= sliding_window_stride){
                    if(new_window_tail == nullptr){
                        new_window_tail = mem_store::create_tail(nullptr);
                        new_window->head = new_window_tail;
                        new_window->end_tail = new_window_tail;
                        new_window->length = 1;
                    } else {
                        tail* old_tail = new_window_tail;
                        new_window_tail = mem_store::create_tail(nullptr);
                        old_tail->set_future(new_window_tail);
                        new_window->length++;
                        new_window->end_tail = new_window_tail;
                    }
                    new_window_tail->tr = new_window;
                    new_window_tail->td = t->td;
                    new_window_tail->split_from = t;
                }
                t = t->future();
                index++;
            }
            tail* old_tail = tr->end_tail;
            tail *end_tail = mem_store::create_tail(nullptr);
            end_tail->td->index = old_tail->get_index() + 1;
            end_tail->tr = tr;
            old_tail->set_future(end_tail);
            tr->end_tail = end_tail;

            traces.push_back(new_window);
        }
    }
}

std::pair<trace*, tail*> inputdata::process_symbol_info(symbol_info &cur_symbol,
                                                        std::unordered_map<std::string, trace*> &trace_map) {
    // Build expected trace / tail strings from symbol info
    auto id = cur_symbol.get_str("id");
    auto symbol = cur_symbol.get_str("symb");
    if (symbol.empty()) symbol = "0";
    auto type = cur_symbol.get_str("type");
    if (type.empty()) type = "0";
    auto trace_attrs = cur_symbol.get("tattr");
    auto symbol_attrs = cur_symbol.get("attr");
    auto data = cur_symbol.get("eval");

    // Get or create the trace for this trace id
    if (!trace_map.contains(id)) {
        auto new_trace = mem_store::create_trace(this);
        new_trace->sequence = this->num_sequences++;
        trace_map.emplace(id, new_trace);
    }
    trace* tr = trace_map.at(id);

    tail* new_tail = make_tail(id, symbol, type, trace_attrs, symbol_attrs, data);

    add_type_to_trace(tr, type, trace_attrs);

    tail* old_tail = tr->end_tail;
    if(old_tail == nullptr){
        tr->head = new_tail;
        tr->end_tail = new_tail;
        tr->length = 1;
        new_tail->tr = tr;
    } else {
        new_tail->td->index = old_tail->get_index() + 1;
        old_tail->set_future(new_tail);
        tr->end_tail = new_tail;
        tr->length++;
        new_tail->tr = tr;
    }

    new_tail->td->index = tr->length - 1;

    return std::make_pair(tr, new_tail);
}


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

tail *inputdata::make_tail(const string& id,
                           const string& symbol,
                           const string& type,
                           const vector<string>& trace_attrs,
                           const vector<string>& symbol_attrs,
                           const vector<string>& data) {

    tail* new_tail = mem_store::create_tail(nullptr);
    tail_data* td = new_tail->td;

    // Add symbol to the alphabet if it isn't in there already
    if(r_alphabet.find(symbol) == r_alphabet.end()){
        r_alphabet[symbol] = (int)alphabet.size();
        alphabet.push_back(symbol);
    }

    // Fill in tail data
    td->symbol = r_alphabet[symbol];
    td->data = strutil::join(data, ",");
    td->tail_nr = num_tails++;

    auto num_symbol_attributes = this->symbol_attributes.size();
    if(num_symbol_attributes > 0){
        for(int i = 0; i < num_symbol_attributes; ++i){
            const string& val = symbol_attrs.at(i);
            td->attr[i] = symbol_attributes[i].get_value(val);
        }
    }

    return new_tail;
}

void inputdata::add_type_to_trace(trace* new_trace,
                                  const string& type,
                                  const vector<string>& trace_attrs) {
    // Add to type map
    if(r_types.find(type) == r_types.end()){
        r_types[type] = (int)types.size();
        types.push_back(type);
    }

    auto num_trace_attributes = this->trace_attributes.size();
    if(num_trace_attributes > 0){
        for(int i = 0; i < num_trace_attributes; ++i){
            const string& val = trace_attrs.at(i);
            new_trace->trace_attr[i] = trace_attributes[i].get_value(val);
        }
    }
    new_trace->type = r_types[type];
}







