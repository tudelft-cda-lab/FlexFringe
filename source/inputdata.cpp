///*
// */
//#include "utility/loguru.hpp"
//
//#include "inputdata.h"
//
//vector<string> inputdata::alphabet;
//map<string, int> inputdata::r_alphabet;
//
//vector<string> inputdata::types;
//map<string, int> inputdata::r_types;
//
//vector<attribute> inputdata::trace_attributes;
//vector<attribute> inputdata::symbol_attributes;
//
///* tail list functions, they are in two lists:
// * 1) past_tail <-> current_tail <-> future_tail
// * 2) split_from <-> current_tail <-> split_to
// * 1 is used to get past and futures from the input data sequences
// * 2 is used to keep track of and be able to quickly undo_merge split refinements
// * */
//void tail::split(tail* t){
//    t->split_from = this;
//    t->future_tail = future_tail;
//    t->past_tail = past_tail;
//    split_to = t;
//}
//
//void tail::undo_split(){
//    split_to = nullptr;
//}
//
//tail* tail::next() const{
//    if(split_to == nullptr) return next_in_list;
//    if(next_in_list == nullptr) return nullptr;
//    return next_in_list->next();
//}
//
//tail* tail::split_to_end(){
//    if(split_to == nullptr) return this;
//    return split_to->split_to_end();
//}
//
//tail* tail::future() const{
//    if(future_tail == nullptr) return nullptr;
//    if(split_to == nullptr) return future_tail->split_to_end();
//    return split_to->future();
//}
//
//tail* tail::past() const{
//    if(past_tail == nullptr) return nullptr;
//    if(split_to == nullptr) return past_tail->split_to_end();
//    return split_to->past();
//}
//
//void tail::set_future(tail* ft){
//    future_tail = ft;
//    ft->past_tail = this;
//}
//
///* tail constructors
// * copies values from existing tail but does not put into any list
// * initialize_after_adding_traces re-used a previously declared but erased tail */
//tail::tail(){
//    td = new tail_data();
//    past_tail = nullptr;
//    future_tail = nullptr;
//    next_in_list = nullptr;
//    split_from = nullptr;
//    split_to = nullptr;
//
//    tr = nullptr;
//}
//
//tail::tail(tail* ot){
//    if(ot != nullptr){
//        td = ot->td;
//        tr = ot->tr;
//    } else {
//        td = new tail_data();
//        tr = nullptr;
//    }
//    past_tail = nullptr;
//    future_tail = nullptr;
//    next_in_list = nullptr;
//    split_from = nullptr;
//    split_to = nullptr;
//}
//
//void tail::initialize(tail* ot){
//    if(ot != nullptr){
//        td = ot->td;
//        tr = ot->tr;
//    } else {
//        td = new tail_data();
//        tr = nullptr;
//    }
//    past_tail = nullptr;
//    future_tail = nullptr;
//    next_in_list = nullptr;
//    split_from = nullptr;
//    split_to = nullptr;
//}
//
///* attribute constructor from string
// * if it contains d -> attribute is discrete
// * if it contains s -> attribute can be used to infer guards
// * if it contains f -> attribute is a distribution variable
// * if it contains t -> attribute is a target variable
// * */
//attribute::attribute(const string& input){
//    discrete = false;
//    splittable = false;
//    distributionable = false;
//    target = false;
//
//    stringstream cs(input);
//    string attr_name;
//    string attr_types;
//    std::getline(cs,attr_name, '=');
//    std::getline(cs,attr_types);
//
//    if(attr_types.find('d') != std::string::npos) discrete = true;
//    if(attr_types.find('s') != std::string::npos) splittable = true;
//    if(attr_types.find('f') != std::string::npos) distributionable = true;
//    if(attr_types.find('t') != std::string::npos) target = true;
//
//    name = attr_name;
//
//    cs.clear();
//}
//
///* inputdata constructor for initialization
// * content is set using the read functions */
//inputdata::inputdata() {
//    num_sequences = 0;
//    max_sequences = 0;
//    node_number = 0;
//    num_tails = 0;
//
//    alphabet.clear();
//    r_alphabet.clear();
//
//    types.clear();
//    r_types.clear();
//
//    trace_attributes.clear();
//    symbol_attributes.clear();
//}
//
//inputdata::~inputdata() {
//    for(auto trace : all_traces){
//        delete trace;
//    }
//}
//
//trace* inputdata::read_csv_row(istream &input_stream) {
//    string line, cell;
//
//    getline(input_stream, line);
//    if(line.empty()) return nullptr;
//
//    stringstream ls2(line);
//    vector<string> row;
//    while (std::getline(ls2, cell, ',')) {
//        row.push_back(cell);
//    }
//    std::getline(ls2, cell);
//    row.push_back(cell);
//
//    string id = "";
//    for (auto i : id_cols) {
//        if (!id.empty()) id.append("__");
//        id.append(row[i]);
//    }
//
//    string type = "";
//    for (auto i : type_cols) {
//        if (!type.empty()) type.append("__");
//        type.append(row[i]);
//    }
//    if(type.empty()) type = "0";
//
//    string symbol = "";
//    for (auto i : symbol_cols) {
//        if (!symbol.empty()) symbol.append("__");
//        symbol.append(row[i]);
//    }
//    if(symbol.empty()) symbol = "0";
//
//    string trace_attr = "";
//    for (auto i : trace_attr_cols){
//        if (!trace_attr.empty()) trace_attr.append(",");
//        trace_attr.append(row[i]);
//    }
//
//    string symbol_attr = "";
//    for (auto i : symbol_attr_cols){
//        if (!symbol_attr.empty()) symbol_attr.append(",");
//        symbol_attr.append(row[i]);
//    }
//
//    string data = "";
//    for (auto i : data_cols) {
//        if (!data.empty()) data.append(",");
//        data.append(row[i]);
//    }
//
//    string abbadingo_type = type;
//    abbadingo_type.append(":" + trace_attr);
//
//    string abbadingo_symbol = symbol;
//    abbadingo_symbol.append(":" + symbol_attr);
//    abbadingo_symbol.append("/" + data);
//
//    auto it = tail_map.find(id);
//    if (it == tail_map.end()) {
//        trace *new_trace = mem_store::create_trace();
//        tail_map.insert(pair<string,trace*>(id,new_trace));
//    }
//    it = tail_map.find(id);
//    trace* tr = it->second;
//    tr->sequence = inputdata::num_sequences++;
//
//    tail *new_tail = mem_store::create_tail(nullptr);
//    istringstream abbadingo_symbol_stream(abbadingo_symbol);
//    read_abbadingo_symbol(abbadingo_symbol_stream, new_tail);
//
//    it = tail_map.find(id);
//    trace* new_tr = it->second;
//    istringstream abbadingo_type_stream(abbadingo_type);
//    read_abbadingo_type(abbadingo_type_stream, new_tr);
//
//    tail* old_tail = new_tr->end_tail;
//    if(old_tail == nullptr){
//        new_tr->head = new_tail;
//        new_tr->end_tail = new_tail;
//        new_tr->length = 1;
//        new_tail->tr = new_tr;
//    } else {
//        new_tr = it->second;
//        old_tail = new_tr->end_tail;
//        new_tail->td->index = old_tail->get_index() + 1;
//        old_tail->set_future(new_tail);
//        new_tr->end_tail = new_tail;
//        new_tr->length++;
//        new_tail->tr = new_tr;
//    }
//
//    if(SLIDING_WINDOW && tr->get_length() == SLIDING_WINDOW_SIZE){
//        if(SLIDING_WINDOW_TYPE){
//            string type_string = inputdata::string_from_symbol(new_tail->get_symbol());
//            if(r_types.find(type_string) == r_types.end()){
//                r_types[type_string] = (int)types.size();
//                types.push_back(type_string);
//            }
//            tr->type = r_types[type_string];
//        }
//        trace* new_window = mem_store::create_trace();
//        new_window->type = tr->type;
//        new_window->sequence = inputdata::num_sequences;
//        tail* t = tr->get_head();
//        int index = 0;
//        tail* new_window_tail = nullptr;
//        while(t != nullptr){
//            if(index >= SLIDING_WINDOW_STRIDE){
//                if(new_window_tail == nullptr){
//                    new_window_tail = mem_store::create_tail(nullptr);
//                    new_window->head = new_window_tail;
//                    new_window->end_tail = new_window_tail;
//                    new_window->length = 1;
//                } else {
//                    tail* old_tail = new_window_tail;
//                    new_window_tail = mem_store::create_tail(nullptr);
//                    old_tail->set_future(new_window_tail);
//                    new_window->length++;
//                    new_window->end_tail = new_window_tail;
//                }
//                new_window_tail->tr = new_window;
//                new_window_tail->td = t->td;
//                new_window_tail->split_from = t;
//            }
//            t = t->future();
//            index++;
//        }
//        tail* old_tail = tr->end_tail;
//        tail *end_tail = mem_store::create_tail(nullptr);
//        end_tail->td->index = old_tail->get_index() + 1;
//        end_tail->tr = tr;
//        old_tail->set_future(end_tail);
//        tr->end_tail = end_tail;
//
//        it->second = new_window;
//    }
//
//    return tr;
//}
//
//void inputdata::read_csv_file(istream &input_stream) {
//    list<trace*> stored_traces;
//    string line, cell;
//
//    while(!input_stream.eof()) {
//        trace *tr = read_csv_row(input_stream);
//
//        if (tr != nullptr && tr->get_end()->is_final()) {
//            //add_trace_to_apta(tr, the_apta);
//            //if (!ADD_TAILS) tr->erase();
//            all_traces.push_back(tr);
//        }
//    }
//
//    if(SLIDING_WINDOW_ADD_SHORTER){
//        while(!tail_map.empty()) {
//            trace *tr = tail_map.begin()->second;
//            tail* old_tail = tr->end_tail;
//            tail *end_tail = mem_store::create_tail(nullptr);
//            end_tail->td->index = old_tail->get_index() + 1;
//            end_tail->tr = tr;
//            old_tail->set_future(end_tail);
//            tr->end_tail = end_tail;
//
//            all_traces.push_back(tr);
//            //add_trace_to_apta(tr, the_apta);
//            tail_map.erase(tail_map.begin());
//            //if (!ADD_TAILS) tr->erase();
//        }
//    }
//}
//
//void inputdata::read_csv_header(istream &input_stream) {
//    string line;
//    getline(input_stream,line);
//    stringstream ls(line);
//    string cell;
//    int index = 0;
//    while(std::getline(ls,cell, ',')){
//        cell.erase(0,cell.find_first_not_of(" \n\r\t"));
//        if(cell.rfind("id", 0) == 0){ id_cols.insert(index); }
//        else if(cell.rfind("type", 0) == 0){ type_cols.insert(index); }
//        else if(cell.rfind("symb", 0) == 0){ symbol_cols.insert(index); }
//        else if(cell.rfind("eval", 0) == 0){ data_cols.insert(index); }
//        else if(cell.rfind("attr", 0) == 0){
//            symbol_attr_cols.insert(index);
//            stringstream cs(cell);
//            string attr;
//            std::getline(cs,attr, ':');
//            std::getline(cs,attr);
//            symbol_attributes.emplace_back(attr);
//            cs.clear();
//        } else if(cell.rfind("tattr", 0) == 0){
//            trace_attr_cols.insert(index);
//            stringstream cs(cell);
//            string attr;
//            std::getline(cs,attr, ':');
//            std::getline(cs,attr);
//            trace_attributes.emplace_back(attr);
//            cs.clear();
//        }
//        else { cerr << "unknown column " << index << endl; }
//
//        index++;
//    }
//}
//
//void trace::reverse(){
//    tail* t = head;
//    while(t != end_tail){
//        tail* temp_future = t->future_tail;
//        t->future_tail = t->past_tail;
//        t->past_tail = temp_future;
//        t = temp_future;
//    }
//    tail* temp_head = head;
//    head = end_tail->past_tail;
//    head->past_tail = nullptr;
//    end_tail->past_tail = temp_head;
//    temp_head->future_tail = end_tail;
//}
//
//
//void inputdata::add_traces_to_apta(apta* the_apta){
//    for(auto* tr : all_traces){
//        add_trace_to_apta(tr, the_apta);
//        if(!ADD_TAILS) tr->erase();
//    }
//}
//
//void inputdata::add_trace_to_apta(trace* tr, apta* the_apta){
//    int depth = 0;
//    apta_node* node = the_apta->root;
//    /*if(node->access_trace == nullptr){
//        node->access_trace = mem_store::create_trace();
//    }*/
//
//    if(REVERSE_TRACES){
//        tr->reverse();
//    }
//
//    tail* t = tr->head;
//
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
//}
//
//void inputdata::read_abbadingo_header(istream &input_stream){
//    input_stream >> max_sequences;
//
//    string tuple;
//    input_stream >> tuple;
//
//    std::stringstream lineStream;
//    lineStream.str(tuple);
//
//    string alph;
//    std::getline(lineStream,alph,':');
//    //alphabet_size = stoi(alph);
//
//    string trace_attr;
//    std::getline(lineStream,trace_attr, ':');
//    string symbol_attr;
//    std::getline(lineStream,symbol_attr);
//    if(symbol_attr.empty()){
//        symbol_attr = trace_attr;
//        trace_attr = "";
//    }
//    lineStream.clear();
//
//    if(!trace_attr.empty()){
//        lineStream.str(trace_attr);
//        string attr;
//        std::getline(lineStream,attr, ',');
//        while(!std::getline(lineStream,attr, ',').eof()){
//            trace_attributes.emplace_back(attr);
//        }
//        trace_attributes.emplace_back(attr);
//        lineStream.clear();
//    }
//
//    if(!symbol_attr.empty()){
//        lineStream.str(symbol_attr);
//        string attr;
//        std::getline(lineStream,attr, ',');
//        while(!std::getline(lineStream,attr, ',').eof()){
//            symbol_attributes.emplace_back(attr);
//        }
//        symbol_attributes.emplace_back(attr);
//        lineStream.clear();
//    }
//
//    lineStream.clear();
//}
//
//void inputdata::read_abbadingo_file(istream &input_stream){
//    for(int line = 0; line < max_sequences; ++line){
//        trace* new_trace = mem_store::create_trace();
//        read_abbadingo_sequence(input_stream, new_trace);
//        new_trace->sequence = line;
//        all_traces.push_back(new_trace);
//        //add_trace_to_apta(new_trace, the_apta);
//        //if(!ADD_TAILS) new_trace->erase();
//    }
//}
//
//void inputdata::read_abbadingo_sequence(istream &input_stream, trace* new_trace){
//    string temp, temp_symbol, data, type_string, type_attr, symbol_string, symbol_attr, val;
//    int length;
//    std::stringstream l1, l2, l3;
//
//    read_abbadingo_type(input_stream, new_trace);
//
//    input_stream >> length;
//    new_trace->length = length;
//
//    tail* new_tail = mem_store::create_tail(nullptr);
//    new_tail->tr = new_trace;
//    new_trace->head = new_tail;
//
//    for(int index = 0; index < length; ++index){
//        read_abbadingo_symbol(input_stream, new_tail);
//        new_tail->td->index = index;
//        tail* old_tail = new_tail;
//        new_tail = mem_store::create_tail(nullptr);
//        new_tail->tr = new_trace;
//        old_tail->set_future(new_tail);
//    }
//    new_tail->td->index = length;
//    new_trace->end_tail = new_tail;
//    new_trace->sequence = inputdata::num_sequences++;
//}
//
//void inputdata::read_abbadingo_type(istream &input_stream, trace* new_trace){
//    string temp, type_string, type_attr, val;
//    std::stringstream l1, l2;
//
//    input_stream >> temp;
//    l1.str(temp);
//    std::getline(l1,type_string,':');
//    std::getline(l1,type_attr);
//    l1.clear();
//
//    if(r_types.find(type_string) == r_types.end()){
//        r_types[type_string] = (int)types.size();
//        types.push_back(type_string);
//    }
//
//    int num_trace_attributes = inputdata::get_num_trace_attributes();
//    if(num_trace_attributes > 0){
//        l2.str(type_attr);
//        for(int i = 0; i < num_trace_attributes; ++i){
//            if(i < num_trace_attributes - 1) std::getline(l2,val,',');
//            else std::getline(l2,val);
//            new_trace->trace_attr[i] = trace_attributes[i].get_value(val);
//        }
//        l2.clear();
//    }
//    new_trace->type = r_types[type_string];
//}
//
//void inputdata::read_abbadingo_symbol(istream &input_stream, tail* new_tail){
//    string temp, temp_symbol, data, type_string, type_attr, symbol_string, symbol_attr, val;
//    std::stringstream l1, l2, l3;
//
//    tail_data* td = new_tail->td;
//
//    input_stream >> std::ws;
//    temp = string(std::istreambuf_iterator<char>(input_stream), {});
//    l1.str(temp);
//    std::getline(l1,temp_symbol,'/');
//    std::getline(l1,data);
//    l1.clear();
//    l2.str(temp_symbol);
//    std::getline(l2,symbol_string,':');
//    std::getline(l2,symbol_attr);
//    l2.clear();
//
//    if(r_alphabet.find(symbol_string) == r_alphabet.end()){
//        r_alphabet[symbol_string] = (int)alphabet.size();
//        alphabet.push_back(symbol_string);
//    }
//
//    td->symbol = r_alphabet[symbol_string];
//    td->data = data;
//    td->tail_nr = num_tails++;
//
//    int num_symbol_attributes = inputdata::get_num_symbol_attributes();
//    if(num_symbol_attributes > 0){
//        l3.str(symbol_attr);
//        for(int i = 0; i < num_symbol_attributes; ++i){
//            if(i < num_symbol_attributes - 1) std::getline(l3,val,',');
//            else std::getline(l3,val);
//            td->attr[i] = symbol_attributes[i].get_value(val);
//        }
//        l3.clear();
//    }
//}
//
//string trace::to_string(){
//    tail* t = head;
//    if(t == nullptr) return "0 0";
//    while(t->future() != nullptr) t = t->future();
//    return "" + inputdata::string_from_type(get_type()) + " " + std::to_string(get_length()) + " " + t->to_string() +"";
//}
//
//string tail::to_string(){
//    ostringstream ostr;
//    tail* t = this;
//    while(t->past() != nullptr) t = t->past();
//
//    while(t != this->future() && !t->is_final()){
//        ostr << inputdata::get_symbol(t->get_symbol());
//        if(inputdata::get_num_symbol_attributes() > 0){
//            ostr << ":";
//            for(int i = 0; i < inputdata::get_num_symbol_attributes(); i++){
//                ostr << t->get_symbol_value(i);
//                if(i + 1 < inputdata::get_num_symbol_attributes())
//                    ostr << ",";
//            }
//        }
//        if(t->get_data() != "") {
//            ostr << "/" << t->get_data();
//        }
//        t = t->future_tail;
//        if(t != this->future_tail && !t->is_final()){
//            ostr << " ";
//        }
//    }
//    return ostr.str();
//}
//
//tail::~tail(){
//    if(split_from == nullptr){
//        delete td;
//    }
//    //if(future_tail != nullptr) delete future_tail;
//}
//
//tail_data::tail_data() {
//    index = -1;
//    symbol = -1;
//    attr = new double[inputdata::get_num_symbol_attributes()];
//    for(int i = 0; i < inputdata::get_num_symbol_attributes(); ++i){
//        attr[i] = 0.0;
//    }
//    data = "";
//    tail_nr = -1;
//}
//tail_data::~tail_data() {
//    delete[] attr;
//}
//
//void tail_data::initialize() {
//    index = -1;
//    symbol = -1;
//    for(int i = 0; i < inputdata::get_num_symbol_attributes(); ++i){
//        attr[i] = 0.0;
//    }
//    data = "";
//    tail_nr = -1;
//}
//
//trace::trace() {
//    sequence = -1;
//    length = -1;
//    type = -1;
//    trace_attr = new double[inputdata::get_num_trace_attributes()];
//    for(int i = 0; i < inputdata::get_num_trace_attributes(); ++i){
//        trace_attr[i] = 0.0;
//    }
//    head = nullptr;
//    end_tail = nullptr;
//    refs = 1;
//}
//
//trace::~trace(){
//    delete trace_attr;
//    delete head;
//}
//
//
//void trace::initialize() {
//    sequence = -1;
//    length = -1;
//    type = -1;
//    for(int i = 0; i < inputdata::get_num_trace_attributes(); ++i){
//        trace_attr[i] = 0.0;
//    }
//    head = nullptr;
//    end_tail = nullptr;
//    refs = 1;
//}
//
//tail* inputdata::access_tail(tail* t) {
//    tail* res = mem_store::create_tail(nullptr);
//    res->td->index = t->td->index;
//    res->td->symbol = t->td->symbol;
//    for(int i = 0; i < inputdata::get_num_symbol_attributes(); ++i){
//        res->td->attr[i] = t->td->attr[i];
//    }
//    res->td->data = t->td->data;
//    return res;
//}
//
//trace* inputdata::access_trace(tail* t){
//    t = t->split_to_end();
//    int length = 1;
//    trace* tr = mem_store::create_trace();
//    tr->sequence = t->tr->sequence;
//    tr->type = t->tr->type;
//    for(int i = 0; i < inputdata::get_num_trace_attributes(); ++i){
//        tr->trace_attr[i] = t->tr->trace_attr[i];
//    }
//    if(STORE_ACCESS_STRINGS){
//        tail* ti = t->tr->head->split_to_end();
//        tail* tir = inputdata::access_tail(ti);
//        tr->head = tir;
//        tir->tr = tr;
//        tail* temp = tr->head;
//        while(ti != t){
//            length++;
//            ti = ti->future();
//            temp = inputdata::access_tail(ti);
//            tir->set_future(temp);
//            tir = temp;
//            tir->tr = tr;
//        }
//        tr->refs = 1;
//        tr->length = length;
//        tr->end_tail = temp;
//    } else {
//        tr->head = inputdata::access_tail(t);
//        tr->refs = 1;
//        tr->length = 1;
//        tr->end_tail = tr->head;
//    }
//    return tr;
//}
//
//void trace::erase(){
//    --refs;
//    if(refs == 0) mem_store::delete_trace(this);
//}
