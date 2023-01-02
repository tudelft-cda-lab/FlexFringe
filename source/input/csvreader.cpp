#include "csvreader.h"
#include "mem_store.h"
#include "stringutil.h"

using namespace std;

void csv_inputdata::read(istream &input_stream) {
    this->readHeader(input_stream);

    while(!input_stream.eof()) {
        trace *tr = this->readRow(input_stream);

        if (tr != nullptr && tr->get_end()->is_final()) {
            traces.push_back(tr);
        }
    }

    while(!trace_map.empty()) {
        trace *tr = trace_map.begin()->second;
        tail* old_tail = tr->end_tail;
        tail* end_tail = mem_store::create_tail(nullptr);
        end_tail->td->index = old_tail->get_index() + 1;
        end_tail->tr = tr;
        old_tail->set_future(end_tail);
        tr->end_tail = end_tail;

        traces.push_back(tr);
        //add_trace_to_apta(tr, the_apta);
        trace_map.erase(trace_map.begin());
        //if (!ADD_TAILS) tr->erase();
    }
}

void csv_inputdata::readHeader(istream &input_stream) {
    string line;
    getline(input_stream,line);
    stringstream ls(line);
    string cell;
    int index = 0;
    while(std::getline(ls,cell, delim)){

        cell.erase(0,cell.find_first_not_of(" \n\r\t"));

        if(cell.rfind("id", 0) == 0){ id_cols.insert(index); }
        else if(cell.rfind("type", 0) == 0){ type_cols.insert(index); }
        else if(cell.rfind("symb", 0) == 0){ symbol_cols.insert(index); }
        else if(cell.rfind("eval", 0) == 0){ data_cols.insert(index); }
        else if(cell.rfind("attr", 0) == 0){
            symbol_attr_cols.insert(index);
            stringstream cs(cell);
            string attr;
            std::getline(cs,attr, ':');
            std::getline(cs,attr);
            symbol_attributes.emplace_back(attr);
            cs.clear();
        } else if(cell.rfind("tattr", 0) == 0){
            trace_attr_cols.insert(index);
            stringstream cs(cell);
            string attr;
            std::getline(cs,attr, ':');
            std::getline(cs,attr);
            trace_attributes.emplace_back(attr);
            cs.clear();
        }
        else { cerr << "unknown column " << index << endl; }

        index++;
    }
}

struct RowData {
    vector<string> id;
    vector<string> type;
    vector<string> symbol;
    vector<string> trace_attr;
    vector<string> symbol_attr;
    vector<string> data;
};

trace* csv_inputdata::readRow(istream &input_stream) {
    string line, cell;
    RowData rowData;

    getline(input_stream, line);
    if(line.empty()) return nullptr;

    stringstream ls2(line);
    vector<string> row;
    while (std::getline(ls2, cell, delim)) {
        if (strip_whitespace) { strutil::trim(cell); }
        row.push_back(cell);
    }
    std::getline(ls2, cell);
    if (strip_whitespace) { strutil::trim(cell); }
    row.push_back(cell);

    string id;
    for (auto i : id_cols) {
        if (!id.empty()) id.append("__");
        id.append(row[i]);
    }

    string type;
    for (auto i : type_cols) {
        if (!type.empty()) type.append("__");
        type.append(row[i]);
    }
    if(type.empty()) type = "0";

    string symbol;
    for (auto i : symbol_cols) {
        if (!symbol.empty()) symbol.append("__");
        symbol.append(row[i]);
    }
    if(symbol.empty()) symbol = "0";

    vector<string> trace_attrs;
    for (auto i : trace_attr_cols){
        trace_attrs.emplace_back(row[i]);
    }

    vector<string> symbol_attrs;
    for (auto i : symbol_attr_cols){
        symbol_attrs.emplace_back(row[i]);
    }

    vector<string> data;
    for (auto i : data_cols) {
        data.emplace_back(row[i]);
    }

//    string abbadingo_type = type;
//    abbadingo_type.append(":" + trace_attr);
//
//    string abbadingo_symbol = symbol;
//    abbadingo_symbol.append(":" + symbol_attr);
//    abbadingo_symbol.append("/" + data);

    auto it = trace_map.find(id);
    if (it == trace_map.end()) {
        trace* new_trace = mem_store::create_trace(dynamic_cast<inputdata *>(this));
        trace_map.insert(pair<string,trace*>(id, new_trace));
    }
    it = trace_map.find(id);
    trace* tr = it->second;
    tr->sequence = this->num_sequences++;

    tail* new_tail = this->make_tail(id, symbol, type, trace_attrs, symbol_attrs, data);

    it = trace_map.find(id);
    trace* new_tr = it->second;
//    istringstream abbadingo_type_stream(abbadingo_type);
//    read_abbadingo_type(abbadingo_type_stream, new_tr);
    this->add_type_to_trace(new_tr, type, trace_attrs);


    tail* old_tail = new_tr->end_tail;
    if(old_tail == nullptr){
        new_tr->head = new_tail;
        new_tr->end_tail = new_tail;
        new_tr->length = 1;
        new_tail->tr = new_tr;
    } else {
        new_tr = it->second;
        old_tail = new_tr->end_tail;
        new_tail->td->index = old_tail->get_index() + 1;
        old_tail->set_future(new_tail);
        new_tr->end_tail = new_tail;
        new_tr->length++;
        new_tail->tr = new_tr;
    }

    if(SLIDING_WINDOW && tr->get_length() == SLIDING_WINDOW_SIZE){
        if(SLIDING_WINDOW_TYPE){
            string type_string = string_from_symbol(new_tail->get_symbol());
            if(r_types.find(type_string) == r_types.end()){
                r_types[type_string] = (int)types.size();
                types.push_back(type_string);
            }
            tr->type = r_types[type_string];
        }
        trace* new_window = mem_store::create_trace(dynamic_cast<inputdata *>(this));
        new_window->type = tr->type;
        new_window->sequence = this->num_sequences;
        tail* t = tr->get_head();
        int index = 0;
        tail* new_window_tail = nullptr;
        while(t != nullptr){
            if(index >= SLIDING_WINDOW_STRIDE){
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
        tail* end_tail = mem_store::create_tail(nullptr);
        end_tail->td->index = old_tail->get_index() + 1;
        end_tail->tr = tr;
        old_tail->set_future(end_tail);
        tr->end_tail = end_tail;

        it->second = new_window;
    }

    return tr;
}

string csv_inputdata::string_from_symbol(int symbol) {
    if(symbol == -1) return "fin";
    if(alphabet.size() < symbol) return "_";
    return alphabet[symbol];
}

// Replaces read_abbadingo_type
void csv_inputdata::add_type_to_trace(trace* new_trace,
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

// replaces read_abbadingo_symbol
tail* csv_inputdata::make_tail(const string& id,
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
    td->data = strutil::join(data, reinterpret_cast<const char *const>(','));
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

const set<int> &csv_inputdata::getIdCols() const {
    return id_cols;
}

const set<int> &csv_inputdata::getTypeCols() const {
    return type_cols;
}

const set<int> &csv_inputdata::getSymbolCols() const {
    return symbol_cols;
}

const set<int> &csv_inputdata::getDataCols() const {
    return data_cols;
}

const set<int> &csv_inputdata::getTraceAttrCols() const {
    return trace_attr_cols;
}

const set<int> &csv_inputdata::getSymbolAttrCols() const {
    return symbol_attr_cols;
}

const vector<attribute> &csv_inputdata::getTraceAttributes() const {
    return trace_attributes;
}

const vector<attribute> &csv_inputdata::getSymbolAttributes() const {
    return symbol_attributes;
}
