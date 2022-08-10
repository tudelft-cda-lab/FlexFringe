#include "csvreader.h"
#include "mem_store.h"

using namespace std;

void CSVInputData::read(istream &input_stream) {
    this->readHeader(input_stream);

    while(!input_stream.eof()) {
        Trace *tr = this->readRow(input_stream);

        if (tr != nullptr && tr->get_end()->is_final()) {
            traces.push_back(tr);
        }
    }

    while(!trace_map.empty()) {
        Trace *tr = trace_map.begin()->second;
        Tail* old_tail = tr->end_tail;
        Tail* end_tail = mem_store::createTail(nullptr);
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

void CSVInputData::readHeader(istream &input_stream) {
    string line;
    getline(input_stream,line);
    stringstream ls(line);
    string cell;
    int index = 0;
    while(std::getline(ls,cell, ',')){
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

Trace* CSVInputData::readRow(istream &input_stream) {
    string line, cell;
    RowData rowData;

    getline(input_stream, line);
    if(line.empty()) return nullptr;

    stringstream ls2(line);
    vector<string> row;
    while (std::getline(ls2, cell, ',')) {
        row.push_back(cell);
    }
    std::getline(ls2, cell);
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

    string trace_attr;
    for (auto i : trace_attr_cols){
        if (!trace_attr.empty()) trace_attr.append(",");
        trace_attr.append(row[i]);
    }

    string symbol_attr;
    for (auto i : symbol_attr_cols){
        if (!symbol_attr.empty()) symbol_attr.append(",");
        symbol_attr.append(row[i]);
    }

    string data;
    for (auto i : data_cols) {
        if (!data.empty()) data.append(",");
        data.append(row[i]);
    }

    string abbadingo_type = type;
    abbadingo_type.append(":" + trace_attr);

    string abbadingo_symbol = symbol;
    abbadingo_symbol.append(":" + symbol_attr);
    abbadingo_symbol.append("/" + data);

    auto it = trace_map.find(id);
    if (it == trace_map.end()) {
        Trace* new_trace = mem_store::createTrace(dynamic_cast<IInputData *>(this));
        trace_map.insert(pair<string,Trace*>(id,new_trace));
    }
    it = trace_map.find(id);
    Trace* tr = it->second;
    tr->sequence = this->num_sequences++;

    Tail* new_tail = mem_store::createTail(nullptr);
    istringstream abbadingo_symbol_stream(abbadingo_symbol);
    read_abbadingo_symbol(abbadingo_symbol_stream, new_tail);

    it = trace_map.find(id);
    Trace* new_tr = it->second;
    istringstream abbadingo_type_stream(abbadingo_type);
    read_abbadingo_type(abbadingo_type_stream, new_tr);

    Tail* old_tail = new_tr->end_tail;
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
        Trace* new_window = mem_store::createTrace(dynamic_cast<IInputData *>(this));
        new_window->type = tr->type;
        new_window->sequence = this->num_sequences;
        Tail* t = tr->get_head();
        int index = 0;
        Tail* new_window_tail = nullptr;
        while(t != nullptr){
            if(index >= SLIDING_WINDOW_STRIDE){
                if(new_window_tail == nullptr){
                    new_window_tail = mem_store::createTail(nullptr);
                    new_window->head = new_window_tail;
                    new_window->end_tail = new_window_tail;
                    new_window->length = 1;
                } else {
                    Tail* old_tail = new_window_tail;
                    new_window_tail = mem_store::createTail(nullptr);
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
        Tail* old_tail = tr->end_tail;
        Tail* end_tail = mem_store::createTail(nullptr);
        end_tail->td->index = old_tail->get_index() + 1;
        end_tail->tr = tr;
        old_tail->set_future(end_tail);
        tr->end_tail = end_tail;

        it->second = new_window;
    }

    return tr;
}

string CSVInputData::string_from_symbol(int symbol) {
    if(symbol == -1) return "fin";
    if(alphabet.size() < symbol) return "_";
    return alphabet[symbol];
}

void CSVInputData::read_abbadingo_type(istream &input_stream, Trace* new_trace){
    string temp, type_string, type_attr, val;
    std::stringstream l1, l2;

    input_stream >> temp;
    l1.str(temp);
    std::getline(l1,type_string,':');
    std::getline(l1,type_attr);
    l1.clear();

    if(r_types.find(type_string) == r_types.end()){
        r_types[type_string] = (int)types.size();
        types.push_back(type_string);
    }

    auto num_trace_attributes = this->trace_attributes.size();
    if(num_trace_attributes > 0){
        l2.str(type_attr);
        for(int i = 0; i < num_trace_attributes; ++i){
            if(i < num_trace_attributes - 1) std::getline(l2,val,',');
            else std::getline(l2,val);
            new_trace->trace_attr[i] = trace_attributes[i].get_value(val);
        }
        l2.clear();
    }
    new_trace->type = r_types[type_string];
}

void CSVInputData::read_abbadingo_symbol(istream &input_stream, Tail* new_tail){
    string temp, temp_symbol, data, type_string, type_attr, symbol_string, symbol_attr, val;
    std::stringstream l1, l2, l3;

    TailData* td = new_tail->td;

    input_stream >> std::ws;
    temp = string(std::istreambuf_iterator<char>(input_stream), {});
    l1.str(temp);
    std::getline(l1,temp_symbol,'/');
    std::getline(l1,data);
    l1.clear();
    l2.str(temp_symbol);
    std::getline(l2,symbol_string,':');
    std::getline(l2,symbol_attr);
    l2.clear();

    if(r_alphabet.find(symbol_string) == r_alphabet.end()){
        r_alphabet[symbol_string] = (int)alphabet.size();
        alphabet.push_back(symbol_string);
    }

    td->symbol = r_alphabet[symbol_string];
    td->data = data;
    td->tail_nr = num_tails++;

    auto num_symbol_attributes = this->symbol_attributes.size();
    if(num_symbol_attributes > 0){
        l3.str(symbol_attr);
        for(int i = 0; i < num_symbol_attributes; ++i){
            if(i < num_symbol_attributes - 1) std::getline(l3,val,',');
            else std::getline(l3,val);
            td->attr[i] = symbol_attributes[i].get_value(val);
        }
        l3.clear();
    }
}

const set<int> &CSVInputData::getIdCols() const {
    return id_cols;
}

const set<int> &CSVInputData::getTypeCols() const {
    return type_cols;
}

const set<int> &CSVInputData::getSymbolCols() const {
    return symbol_cols;
}

const set<int> &CSVInputData::getDataCols() const {
    return data_cols;
}

const set<int> &CSVInputData::getTraceAttrCols() const {
    return trace_attr_cols;
}

const set<int> &CSVInputData::getSymbolAttrCols() const {
    return symbol_attr_cols;
}

const vector<Attribute> &CSVInputData::getTraceAttributes() const {
    return trace_attributes;
}

const vector<Attribute> &CSVInputData::getSymbolAttributes() const {
    return symbol_attributes;
}
