//
// Created by tom on 1/4/23.
//

#include "csvparser.h"

void csv_parser::parse() {

}

void csv_parser::step() {

}

csv_header_parser::csv_header_parser(const std::vector<std::string> &headers) {
    std::set<std::string> default_col_type_names({
        "id"
    })
    for(auto header: headers) {

    }
}

csv_header_parser::csv_header_parser(const std::vector<std::string> &headers,
                                     const std::vector<std::string> &col_type_names) {

}


const std::set<int>& csv_header_parser::get(const std::string& type) const {
    return col_types.at(type);
}

