//
// Created by tom on 1/6/23.
//

#include "symbol_info.h"

const std::vector<std::string> &symbol_info::get(const std::string &name) const {
    return properties.at(name);
}

void symbol_info::set(const std::string &name, const std::vector<std::string> &property_list) {
    properties.emplace(name, property_list);
}

void symbol_info::set(const std::string &name, const std::string &property) {
    properties.emplace(name, std::vector<std::string> {property});
}
