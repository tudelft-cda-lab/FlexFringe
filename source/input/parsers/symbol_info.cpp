//
// Created by tom on 1/6/23.
//

#include "symbol_info.h"
#include "stringutil.h"
#include <algorithm>

const std::vector<std::string> &symbol_info::get(const std::string &name) {
        auto idx = get_property_index(name);
        return properties[idx];
}

void symbol_info::set(const std::string &name, const std::vector<std::string> &property_list) {
    auto idx = get_property_index(name);
    properties[idx] = {property_list.begin(), property_list.end()};
}

void symbol_info::set(const std::string &name, const std::string &property) {
    auto idx = get_property_index(name);
    properties[idx] = { property };
}

std::string symbol_info::get_str(const std::string &name) {
    return strutil::join(get(name), "__");
}

bool symbol_info::has(const std::string &name) {
    auto idx = get_property_index(name);
    auto property = properties[idx];
    return !property.empty();
}

ssize_t symbol_info::get_property_index(const std::string &name) {
    auto idx = std::distance(symbol_info::property_types.begin(),
                         std::find(symbol_info::property_types.begin(),
                                   symbol_info::property_types.end(),
                                   name));

    if (idx > symbol_info::property_types.size()) {
        // TODO: Properly handle this
        abort();
    }

    return idx;
}

