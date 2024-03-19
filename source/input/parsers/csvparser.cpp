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

    if (!row_read) {
        return std::nullopt;
    }

    symbol_info cur_symbol;

    // Handle everything besides attr and tattr columns
    for (const auto &label: header_parser->get_non_reserved_column_type_names()) {
        cur_symbol.set(label, get_vec_from_row(label, row));
    }

    // Handle attr columns
    size_t idx{};
    auto attr_col_names = header_parser->get_names("attr");
    for (auto col_idx: header_parser->get("attr")) {
        cur_symbol.push_symb_attr_info({attr_col_names.at(idx),
                                        row[col_idx].get(),
                                        header_parser->get_col_attr_types(col_idx)});
        idx++;
    }

    // Handle tattr columns
    // Do we have trace attribute info for this trace ID?
    auto cur_trace_id = cur_symbol.get_str("id");
    if (!tattr_info.contains(cur_trace_id)) {
        tattr_info.insert(std::make_pair<>(cur_trace_id, std::make_shared<std::vector<attribute_info>>()));
    }
    auto cur_trace_attr_info = tattr_info.at(cur_trace_id);
    cur_symbol.set_trace_attr_info(cur_trace_attr_info);
    // Fill in the trace attributes for this symbol
    idx = 0;
    auto tattr_col_names = header_parser->get_names("tattr");
    for (auto col_idx: header_parser->get("tattr")) {
        auto tattr_value = row[col_idx].get();
        if (!tattr_value.empty()) {
            cur_symbol.push_trace_attr_info({tattr_col_names.at(idx),
                                             row[col_idx].get(),
                                             header_parser->get_col_attr_types(col_idx)});
        }
        idx++;
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

// These are special cases, resembling attributes, which need special handling
const std::set<std::string> csv_header_parser::reserved_col_type_names = {
        "attr", "tattr"
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
    for (auto &col_type_name: col_type_names) {
        col_types.emplace(col_type_name, std::set<int>{});
        col_names.emplace(col_type_name, std::vector<std::string>{});
    }
}

void csv_header_parser::parse(const std::vector<std::string> &headers) {
    // The type names that indicate a column containing trace or symbol attributes
    const std::set<std::string> type_name_attrs = reserved_col_type_names;

    // The type names that indicate a column contains other relevant information
    std::set<std::string> type_names;
    std::set_difference(col_type_names.begin(), col_type_names.end(),
                        type_name_attrs.begin(), type_name_attrs.end(),
                        std::inserter(type_names, type_names.begin()));

    int idx = 0;
    for (const auto &header: headers) {

        // Parse the current column header with lexy
        auto input = lexy::string_input(header);
        auto result = lexy::parse<csv_header_grammar::col_name>(input, lexy_ext::report_error);
        if (!result.has_value()) {
            throw std::runtime_error(fmt::format("Error parsing column header from column {} - {}", idx, header));
        }
        const auto &parsed_header = result.value();

        // CASE 1: If only a name is specified, we check if it's a valid column type name
        if (!parsed_header.type_name.has_value() && !parsed_header.attr_types.has_value()) {

            // If its attr or tattr, we can't parse it without additional information
            if (type_name_attrs.contains(parsed_header.name)) {
                throw std::runtime_error(fmt::format("Error parsing column header from column {} - {}", idx, header));
            }

            // Otherwise, we try our best (if the column name is a column type)
            if (type_names.contains(parsed_header.name)) {
                col_types.at(header).emplace(idx);
                col_names.at(header).emplace_back(header);
            }
            idx++;
            continue;
        }

        // CASE 2: Do we have a name and a column type? (col_type:col_name)
        if (parsed_header.type_name.has_value() && !parsed_header.attr_types.has_value()) {
            const std::string &type = parsed_header.type_name.value();
            const std::string &name = parsed_header.name;
            col_types.at(type).emplace(idx);
            col_names.at(type).emplace_back(name);
        }

        // CASE 3: We have a trace or symbol attribute column ({attr,tattr}/{d,s,f,t}+:col_name)
        if (parsed_header.type_name.has_value() && parsed_header.attr_types.has_value()) {
            const std::string &type = parsed_header.type_name.value();
            const std::string &name = parsed_header.name;
            const std::set<std::string> &attr_type = parsed_header.attr_types.value();

            std::set<char> attr_type_char;
            for (auto &t: attr_type) {
                attr_type_char.insert(*t.c_str());
            }

            col_types.at(type).emplace(idx);
            col_names.at(type).emplace_back(name);
            attr_types.insert(std::make_pair<>(idx, attr_type_char));
        }
        idx++;
    }

    // Verify that the symbol and trace attribute names are unique
    check_duplicates("attr");
    check_duplicates("tattr");
}

void csv_header_parser::check_duplicates(const std::string &col_type) const {
    if (!col_names.contains(col_type)) { return; }
    std::vector<std::string> attr_names = col_names.at(col_type);
    std::sort(attr_names.begin(), attr_names.end());
    auto duplicate = std::adjacent_find(attr_names.begin(), attr_names.end());
    if (duplicate != attr_names.end()) {
        throw std::runtime_error(fmt::format("Duplicate attribute name: {}", *duplicate));
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

std::set<std::string> csv_header_parser::get_non_reserved_column_type_names() {
    if (!non_reserved_col_type_names.has_value()) {
        non_reserved_col_type_names = std::set<std::string>{};
        std::set_difference(col_type_names.begin(), col_type_names.end(),
                            reserved_col_type_names.begin(), reserved_col_type_names.end(),
                            std::inserter(
                                    non_reserved_col_type_names.value(),
                                    non_reserved_col_type_names.value().begin()
                            ));
    }
    return non_reserved_col_type_names.value();
}

const std::set<char> &csv_header_parser::get_col_attr_types(size_t idx) const {
    return attr_types.at(idx);
}


