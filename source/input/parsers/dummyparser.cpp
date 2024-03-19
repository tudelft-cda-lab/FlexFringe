//
// Created by tom on 1/30/23.
//

#include "input/parsers/dummyparser.h"

std::optional<symbol_info> dummyparser::next() {
    if (symbols.empty()) {
        return std::nullopt;
    }

    const auto tmp = symbols.front();
    symbols.pop_front();
    return tmp;
}

dummyparser &dummyparser::add(const symbol_info &s) {
    symbols.push_back(s);
    return *this;
}
