//
// Created by tom on 1/4/23.
//

#include "csvparser.h"
#include "stringutil.h"
#include "mem_store.h"

void csv_parser::parse(inputdata *pInputdata) {
    for (csv::CSVRow &row: *reader) {
        ID id = get_str_from_row("id", row);

        std::string type = get_str_from_row("type", row);
        if (type.empty()) type = "0";

        std::string symbol = get_str_from_row("symbol", row);
        if (symbol.empty()) symbol = "0";

        std::vector<std::string> trace_attrs = get_vec_from_row("tattr", row);
        std::vector<std::string> symbol_attrs = get_vec_from_row("attr", row);
        std::vector<std::string> data = get_vec_from_row("eval", row);

        trace* tr = get_or_create_trace(id, pInputdata);
        pInputdata->num_sequences
    }
}

std::string csv_parser::get_str_from_row(const std::string &label, const csv::CSVRow &row) {
    std::string result;
    for (auto i: header_parser->get(label)) {
        if (!result.empty()) {
            result.append("__");
        }
        result.append(row[i].get());
    }
    return result;
}

std::vector<std::string> csv_parser::get_vec_from_row(const std::string &label, const csv::CSVRow &row) {
    std::vector<std::string> result;
    for (auto i: header_parser->get(label)) {
        result.emplace_back(row[i].get());
    }
    return result;
}

trace *csv_parser::make_tail(const std::string &id,
                             const std::string &symbol,
                             const std::string &type,
                             const std::vector<std::string> &trace_attrs,
                             const std::vector<std::string> &symbol_attrs,
                             const std::vector<std::string> &data) {
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

std::vector<trace *> csv_parser::get_traces() {
    std::vector<trace *> result(trace_map.size());
    for (const auto & [id, trace]: trace_map) {
        result.push_back(trace);
    }
    return result;
}

trace *csv_parser::get_or_create_trace(std::string id, inputdata* inputData) {
    if (!trace_map.contains(id)) {
        trace* new_trace = mem_store::create_trace(inputData);
        trace_map.insert(std::make_pair(id, new_trace));
    }
    return trace_map.at(id);
}


const std::set<std::string> csv_header_parser::default_col_type_names = {
        "id", "type", "symb", "eval", "attr", "tattr"
};

csv_header_parser::csv_header_parser(const std::vector<std::string> &headers) {
    col_type_names = default_col_type_names;
    setup_col_maps();
    parse(headers);
}

csv_header_parser::csv_header_parser(const std::vector<std::string> &headers,
                                     const std::set<std::string> &col_type_names) {
    this->col_type_names = col_type_names;
    setup_col_maps();
    parse(headers);
}

void csv_header_parser::setup_col_maps() {
    for (const auto &col_type_name: col_type_names) {
        col_types.emplace(col_type_name, std::set<int>{});
        col_names.emplace(col_type_name, std::vector<std::string>{});
    }
}

void csv_header_parser::parse(const std::vector<std::string> &headers) {
    int idx = 0;
    for (auto header: headers) {
        // Do we have a : ?
        auto delim_pos = header.find(':');

        // If there is no delimiter, skip this header
        if (delim_pos == std::string::npos) {
            idx++;
            continue;
        }

        // Get the type name and the col name
        std::string type = header.substr(0, delim_pos);
        std::string name = header.substr(delim_pos + 1);

        // Add the current column idx to the corresponding type idx list
        if (!col_types.contains(type)) {
            col_types.emplace(type, std::set<int>{idx});
        } else {
            col_types.at(type).emplace(idx);
        }

        // Add the names of the columns to the corresponding attribute map
        if (!col_names.contains(type)) {
            col_names.emplace(type, std::vector<std::string>{name});
        } else {
            col_names.at(type).emplace_back(name);
        }

        idx++;
    }
}

const std::set<int> &csv_header_parser::get(const std::string &type) const {
    return col_types.at(type);
}

const std::vector<std::string> &csv_header_parser::get_names(const std::string &type) const {
    return col_names.at(type);
}

const std::set<std::string> &csv_header_parser::get_column_type_names() const {
    return col_type_names;
}




