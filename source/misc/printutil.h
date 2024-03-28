/**
 * @file misc/printutil.h
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief Overload << for easier printing
 * @version 0.1
 * @date 2023-04-06
 *
 * @copyright Copyright (c) 2023
 *
 * This file provides some overloads for << to provide easier printing with containers.
 * If your containers have objects that also can be printed, it works recursive!
 * For use with utility/loguru.hpp include misc/printutil.h first (automatic if sorting includes alphabetically).
 * Finally it adds utils::tostr which can be used to convert everything that could be used with << to a string directly.
 */

#ifndef _PRINTUTIL_H_
#define _PRINTUTIL_H_

#include "utils.h"
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

template <typename T> std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
    o << "[";
    if (v.empty()) {
        o << "]";
        return o;
    }
    // For every item except the last write "Item, "
    for (auto it = v.begin(); it != --v.end(); it++) { o << *it << ", "; }
    // Write out the last item
    o << v.back() << "]";
    return o;
}

template <typename T> std::ostream& operator<<(std::ostream& o, const std::set<T>& v) {
    o << "{";
    if (v.size() == 1) {
        o << "}";
        return o;
    }
    // For every item except the last write "Item, "
    for (auto it : v) {
        o << it;
        if (it != *v.rbegin())
            o << ", ";
    }
    // Write out the last item
    o << "}";
    return o;
}

template <typename T, typename S> std::ostream& operator<<(std::ostream& os, const std::pair<T, S>& v) {
    os << "(";
    os << v.first << ", " << v.second << ")";
    return os;
}

template <typename KeyT, typename ValueT> std::ostream& operator<<(std::ostream& o, const std::map<KeyT, ValueT>& m) {
    o << "{";
    if (m.empty()) {
        o << "}";
        return o;
    }
    // For every pair except the last write "Key: Value, "
    for (auto it = m.begin(); it != --m.end(); it++) {
        const auto& [key, value] = *it;
        o << key << ": " << value << ", ";
    }
    // Write out the last item
    const auto& [key, value] = *--m.end();
    o << key << ": " << value << "}";
    return o;
}
template <typename KeyT, typename ValueT>
std::ostream& operator<<(std::ostream& o, const std::unordered_map<KeyT, ValueT>& m) {
    o << "{";
    if (m.empty()) {
        o << "}";
        return o;
    }
    // For every pair except the last write "Key: Value, "
    for (auto it = m.begin(); it != --m.end(); it++) {
        const auto& [key, value] = *it;
        o << key << ": " << value << ", ";
    }
    // Write out the last item
    const auto& [key, value] = *--m.end();
    o << key << ": " << value << "}";
    return o;
}

namespace utils {
template <class... Args> std::string tostr(Args&&... args) {
    std::ostringstream ostr;
    (ostr << ... << args);
    return ostr.str();
}
} // namespace utils
#endif
