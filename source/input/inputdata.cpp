#include "input/inputdata.h"
#include "apta.h"
#include "stringutil.h"

using namespace std;

/**
 * Read symbols from a parser and construct inputdata from them.
 * @param input_parser the input parser to read from
 */
void inputdata::read(parser *input_parser) {
    auto strategy = read_all();
    auto next_trace_maybe = read_trace(*input_parser, strategy);

    // Loop through all the traces, they are added and finalized automatically
    while (next_trace_maybe.has_value()) {
        auto next_trace = next_trace_maybe.value();
        next_trace_maybe = read_trace(*input_parser, strategy);
    }

    // Make sure they are ordered by sequence number
    traces.sort([](auto &left, auto &right) {
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
void inputdata::read_slidingwindow(parser *input_parser,
                                   ssize_t sliding_window_size,
                                   ssize_t sliding_window_stride,
                                   bool sliding_window_type) {

    auto strategy = slidingwindow(sliding_window_size,
                                  sliding_window_stride,
                                  sliding_window_type);
    auto next_trace_maybe = read_trace(*input_parser, strategy);

    // Loop through all the traces, they are added and finalized automatically
    while (next_trace_maybe.has_value()) {
        auto next_trace = next_trace_maybe.value();
        next_trace_maybe = read_trace(*input_parser, strategy);
    }

    // Make sure they are ordered by sequence number
    traces.sort([](auto &left, auto &right) {
        return left->sequence < right->sequence;
    });
}

std::pair<trace *, tail *> inputdata::process_symbol_info(symbol_info &cur_symbol,
                                                          std::unordered_map<std::string, trace *> &trace_map) {
    // Build expected trace / tail strings from symbol info
    auto id = cur_symbol.get_str("id");
    auto symbol = cur_symbol.get_str("symb");
    auto type = cur_symbol.get_str("type");
    if (type.empty()) type = "0";
    auto data = cur_symbol.get("eval");

    // Get or create the trace for this trace id
    if (!trace_map.contains(id)) {
        auto new_trace = mem_store::create_trace(this);
        new_trace->sequence = this->num_sequences++;
        trace_map.emplace(id, new_trace);
    }

    trace *tr = trace_map.at(id);
    process_trace_attributes(cur_symbol, tr);

    add_type_to_trace(tr, type);

    tail *new_tail = make_tail(symbol, data);
    process_symbol_attributes(cur_symbol, new_tail);

    tail *old_tail = tr->end_tail;
    if (old_tail == nullptr) {
        tr->head = new_tail;
        tr->end_tail = new_tail;
        tr->length = symbol.empty() ? 0 : 1;
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
    if (attr < trace_attributes.size()) {
        return &trace_attributes[attr];
    }
    return nullptr;
}

attribute *inputdata::get_symbol_attribute(int attr) {
    if (attr < symbol_attributes.size()) {
        return &symbol_attributes[attr];
    }
    return nullptr;
}

attribute *inputdata::get_attribute(int attr) {
    if (attr < symbol_attributes.size()) {
        return &symbol_attributes[attr];
    }
    attr = attr - symbol_attributes.size();
    if (attr < trace_attributes.size()) {
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
    if (r_alphabet.find(symbol) == r_alphabet.end()) {
        r_alphabet[symbol] = alphabet.size();
        alphabet.push_back(symbol);
    }
    return r_alphabet[symbol];
}

std::string inputdata::string_from_symbol(int symbol) {
    if (symbol == -1) return "fin";
    if (alphabet.size() < symbol) return "_";
    return alphabet[symbol];
}

int inputdata::type_from_string(std::string type) {
    if (r_types.find(type) == r_types.end()) {
        r_types[type] = types.size();
        types.push_back(type);
    }
    return r_types[type];
}

std::string inputdata::string_from_type(int type) {
    return types[type];
}

void inputdata::add_traces_to_apta(apta *the_apta) {
    for (auto *tr: traces) {
        add_trace_to_apta(tr, the_apta);
        if (!ADD_TAILS) tr->erase();
    }
}

void inputdata::add_trace_to_apta(trace *tr, apta *the_apta) {
    int depth = 0;
    apta_node *node = the_apta->root;
    /*if(node->access_trace == nullptr){
        node->access_trace = mem_store::create_trace();
    }*/

    if (REVERSE_TRACES) {
        tr->reverse();
    }

    tail *t = tr->head;

    while (t != nullptr) {
        node->size = node->size + 1;
        node->add_tail(t);
        node->data->add_tail(t);

        depth++;
        if (t->is_final()) {
            node->final = node->final + 1;
        } else {
            int symbol = t->get_symbol();
            if (node->child(symbol) == nullptr) {
                if (node->size < PARENT_SIZE_THRESHOLD) {
                    break;
                }
                auto *next_node = mem_store::create_node(nullptr);
                node->set_child(symbol, next_node);
                next_node->source = node;
                //next_node->access_trace = inputdata::access_trace(t);
                next_node->depth = depth;
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
    trace *tr = mem_store::create_trace(this);
    tr->sequence = t->tr->sequence;
    tr->type = t->tr->type;
    for (int i = 0; i < this->get_num_trace_attributes(); ++i) {
        tr->trace_attr[i] = t->tr->trace_attr[i];
    }
    if (STORE_ACCESS_STRINGS) {
        tail *ti = t->tr->head->split_to_end();
        tail *tir = this->access_tail(ti);
        tr->head = tir;
        tir->tr = tr;
        tail *temp = tr->head;
        while (ti != t) {
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
    tail *res = mem_store::create_tail(nullptr);
    res->td->index = t->td->index;
    res->td->symbol = t->td->symbol;
    for (int i = 0; i < this->get_num_symbol_attributes(); ++i) {
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

tail *inputdata::make_tail(const string &symbol,
                           const vector<string> &data) {

    tail *new_tail = mem_store::create_tail(nullptr);
    auto td = new_tail->td;

    if (!symbol.empty()) {
        // Add symbol to the alphabet if it isn't in there already
        if (r_alphabet.find(symbol) == r_alphabet.end()) {
            r_alphabet[symbol] = (int) alphabet.size();
            alphabet.push_back(symbol);
        }

        // Fill in tail data
        td->symbol = r_alphabet[symbol];
        td->data = strutil::join(data, ",");
        td->tail_nr = num_tails++;
    }

    return new_tail;
}

void inputdata::add_type_to_trace(trace *new_trace,
                                  const string &type) {
    // Add to type map
    if (r_types.find(type) == r_types.end()) {
        r_types[type] = (int) types.size();
        types.push_back(type);
    }
    new_trace->type = r_types[type];
}

void inputdata::process_trace_attributes(symbol_info &symbolinfo, trace *tr) {
    auto trace_id = symbolinfo.get_str("id");
    if (processed_trace_ids.contains(trace_id)) {
        return;
    }

    // Add attributes to the attribute list first if neccessary
    // TODO: allow dynamically adding attributes?
    // TODO: would probably need a map with attr name -> attr info
    auto trace_attribute_info = symbolinfo.get_trace_attr_info();

    // Do we even have tattr info?
    // TODO: figure out how to arrange this for csv parsing as well
    if (trace_attribute_info == nullptr) {
        return;
    }

    if (trace_attributes.empty() && trace_attributes.size() < trace_attribute_info->size()) {
        for (auto &tattr_info: *trace_attribute_info) {
            trace_attributes.emplace_back(tattr_info);
        }
    }

    // Add the actual values of the attributes too
    size_t idx{};
    for (auto &tattr_info: *trace_attribute_info) {
        tr->trace_attr[idx] = trace_attributes[idx].get_value(tattr_info.get_value());
        idx++;
    }

    processed_trace_ids.insert(trace_id);
}

void inputdata::process_symbol_attributes(symbol_info &symbolinfo, tail *t) {
    auto symbol_attribute_info = symbolinfo.get_symb_attr_info();
    if (symbol_attributes.empty() && symbol_attributes.size() < symbol_attribute_info.size()) {
        for (auto &sattr_info: symbol_attribute_info) {
            symbol_attributes.emplace_back(sattr_info);
        }
    }

    size_t idx{};
    for (auto &sattr_info: symbol_attribute_info) {
        t->td->attr[idx] = symbol_attributes[idx].get_value(sattr_info.get_value());
        idx++;
    }
}

/**
 * Reads a trace from the given input parser according to the provided strategy
 * @param input_parser input parser to read from
 * @param strategy strategy to follow for trace building
 * @return an optional trace pointer. If it is nullopt, there were not enough symbols to build a trace.
 */
std::optional<trace *> inputdata::read_trace(parser &input_parser, reader_strategy &strategy) {
    auto tr_maybe = strategy.read(input_parser, *this);

    if (tr_maybe.has_value()) {
        auto tr = tr_maybe.value();
        tr->finalize();
        traces.push_back(tr);
    }

    return tr_maybe;
}

inputdata inputdata::with_alphabet_from(inputdata &other) {
    inputdata new_inputdata;

    new_inputdata.alphabet = other.alphabet;
    new_inputdata.r_alphabet = other.r_alphabet;
    new_inputdata.types = other.types;
    new_inputdata.r_types = other.r_types;

    return new_inputdata;
}







