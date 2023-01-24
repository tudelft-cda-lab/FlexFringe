//
// Created by tom on 1/4/23.
//

#include "csvparser.h"
#include "stringutil.h"
#include "mem_store.h"
#include "lexy/action/parse.hpp"
#include "input/parsers/grammar/csvheader.h"
#include "lexy_ext/report_error.hpp"
#include "lexy/input/string_input.hpp"
#include "fmt/format.h"

std::optional<symbol_info> csv_parser::next() {
    csv::CSVRow row;
    bool row_read = reader->read_row(row);

    if(!row_read) {
        return std::nullopt;
    }

    symbol_info cur_symbol;

    for (auto label: header_parser->get_column_type_names()) {
        cur_symbol.set(label, get_vec_from_row(label, row));
    }

    return cur_symbol;
}

[[maybe_unused]] std::string csv_parser::get_str_from_row(const std::string &label, const csv::CSVRow &row) {
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
    for (auto& col_type_name: col_type_names) {
        col_types.emplace(col_type_name, std::set<int>{});
        col_names.emplace(col_type_name, std::vector<std::string>{});
    }
}

void csv_header_parser::parse(const std::vector<std::string> &headers) {
    int idx = 0;
    for (const auto& header: headers) {
        auto input = lexy::string_input(header);

        auto result = lexy::parse<csv_header_grammar::col_name>(input, lexy_ext::report_error);
        if (!result.has_value()) {
            throw std::runtime_error(fmt::format("Error parsing column header from column {} - {}", idx, header));
        }

        const auto& parsed_header = result.value();

        // Do we have a : ?
        auto delim_pos = header.find(':');

        // If there is no delimiter, skip this header
        if (delim_pos == std::string::npos) {
            // Unless the header itself is a label
            if (col_type_names.contains(header)) {
                col_types.at(header).emplace(idx);
                col_names.at(header).emplace_back(header);
            }
            idx++;
            continue;
        }

        // Get the type name and the col name
        std::string type = header.substr(0, delim_pos);
        std::string name = header.substr(delim_pos + 1);

        col_types.at(type).emplace(idx);
        col_names.at(type).emplace_back(name);

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




