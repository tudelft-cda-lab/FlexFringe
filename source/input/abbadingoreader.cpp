#include "abbadingoreader.h"
#include "mem_store.h"

void abbadingo_inputdata::read(std::istream &input_stream) {
    read_abbadingo_header(input_stream);

    for(int line = 0; line < max_sequences; ++line){
        trace* new_trace = mem_store::create_trace(this);
        read_abbadingo_sequence(input_stream, new_trace);
        new_trace->sequence = line;
        traces.push_back(new_trace);
    }
}

void abbadingo_inputdata::read_abbadingo_sequence(std::istream &input_stream, trace* new_trace) {
    std::string temp, temp_symbol, data, type_string, type_attr, symbol_string, symbol_attr, val;
    int length;
    std::stringstream l1, l2, l3;

    read_abbadingo_type(input_stream, new_trace);

    input_stream >> length;
    new_trace->length = length;

    tail* new_tail = mem_store::create_tail(nullptr);
    new_tail->tr = new_trace;
    new_trace->head = new_tail;

    for(int index = 0; index < length; ++index){
        read_abbadingo_symbol(input_stream, new_tail);
        new_tail->td->index = index;
        tail* old_tail = new_tail;
        new_tail = mem_store::create_tail(nullptr);
        new_tail->tr = new_trace;
        old_tail->set_future(new_tail);
    }
    new_tail->td->index = length;
    new_trace->end_tail = new_tail;
    new_trace->sequence = num_sequences++;
}

void abbadingo_inputdata::read_abbadingo_type(std::istream &input_stream, trace* new_trace) {
    std::string temp, type_string, type_attr, val;
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

    int num_trace_attributes = inputdata::get_num_trace_attributes();
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

void abbadingo_inputdata::read_abbadingo_symbol(std::istream &input_stream, tail* new_tail) {
    std::string temp, temp_symbol, data, type_string, type_attr, symbol_string, symbol_attr, val;
    std::stringstream l1, l2, l3;

    auto td = new_tail->td;

    input_stream >> temp;
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

    int num_symbol_attributes = inputdata::get_num_symbol_attributes();
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

void abbadingo_inputdata::read_abbadingo_header(std::istream &input_stream) {
    input_stream >> max_sequences;

    std::string tuple;
    input_stream >> tuple;

    std::stringstream lineStream;
    lineStream.str(tuple);

    std::string alph;
    std::getline(lineStream,alph,':');
    //alphabet_size = stoi(alph);

    std::string trace_attr;
    std::getline(lineStream,trace_attr, ':');
    std::string symbol_attr;
    std::getline(lineStream,symbol_attr);
    if(symbol_attr.empty()){
        symbol_attr = trace_attr;
        trace_attr = "";
    }
    lineStream.clear();

    if(!trace_attr.empty()){
        lineStream.str(trace_attr);
        std::string attr;
        std::getline(lineStream,attr, ',');
        while(!std::getline(lineStream,attr, ',').eof()){
            trace_attributes.emplace_back(attr);
        }
        trace_attributes.emplace_back(attr);
        lineStream.clear();
    }

    if(!symbol_attr.empty()){
        lineStream.str(symbol_attr);
        std::string attr;
        std::getline(lineStream,attr, ',');
        while(!std::getline(lineStream,attr, ',').eof()){
            symbol_attributes.emplace_back(attr);
        }
        symbol_attributes.emplace_back(attr);
        lineStream.clear();
    }

    lineStream.clear();
}
