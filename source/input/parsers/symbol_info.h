//
// Created by tom on 1/6/23.
//

#ifndef FLEXFRINGE_SYMBOL_INFO_H
#define FLEXFRINGE_SYMBOL_INFO_H


#include <utility>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <algorithm>
#include <fmt/core.h>
#include "attribute_info.h"

/**
 * This class represent one input symbol, read from an input source.
 * It should contain all the information necessary to later on be turned into
 * tails and traces in some inputdata instance.
 */
class symbol_info {
private:
    std::unordered_map<std::string, std::vector<std::string>> properties;
    std::shared_ptr<std::vector<attribute_info>> trace_attribute_info;
    std::vector<attribute_info> symbol_attribute_info;

public:
    void set(const std::string &name, const std::vector<std::string> &property_list);

    void set(const std::string &name, const std::string &property);

    const std::vector<std::string> &get(const std::string &name);

    std::string get_str(const std::string &name);

    bool has(const std::string &name);

    // Attribute info getters and setters
    void push_trace_attr_info(const attribute_info &tattr) {
        // Do we already have an attribute with this name?
        auto existing_attr = std::find_if(trace_attribute_info->begin(),
                                          trace_attribute_info->end(),
                                          [&tattr](auto &el) {
                                              return el.get_name() == tattr.get_name();
                                          });
        if (existing_attr != trace_attribute_info->end()) {
            throw std::runtime_error(fmt::format(
                    "Error: duplicate trace attribute value \"{}\" specified for trace with id: {}",
                    tattr.get_name(), get_str("id")
                    ));
        }
        else {
            trace_attribute_info->push_back(tattr);
        }
    }

    void set_trace_attr_info(std::shared_ptr<std::vector<attribute_info>> tattr_info) {
        trace_attribute_info = std::move(tattr_info);
    }

    std::shared_ptr<std::vector<attribute_info>> get_trace_attr_info() {
        return trace_attribute_info;
    }

    void push_symb_attr_info(const attribute_info &attr) {
        symbol_attribute_info.push_back(attr);
    }

    const std::vector<attribute_info> &get_symb_attr_info() {
        return symbol_attribute_info;
    }
};


#endif //FLEXFRINGE_SYMBOL_INFO_H
