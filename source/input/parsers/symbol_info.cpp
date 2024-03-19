//
// Created by tom on 1/6/23.
//

#include "symbol_info.h"
#include "stringutil.h"
const std::vector<std::string> &symbol_info::get(const std::string &name) {
    if (!properties.contains(name)){
        properties.emplace(name, std::vector<std::string> {});
    }
    return properties.at(name);
}

void symbol_info::set(const std::string &name, const std::vector<std::string> &property_list) {
    properties.emplace(name, property_list);
}

void symbol_info::set(const std::string &name, const std::string &property) {
    properties.emplace(name, std::vector<std::string> {property});
}

std::string symbol_info::get_str(const std::string &name) {
    return strutil::join(get(name), "__");
}

bool symbol_info::has(const std::string &name) {
    return properties.contains(name) && !properties.at(name).empty();
}
