//
// Created by tom on 1/4/23.
//

#ifndef FLEXFRINGE_CSVPARSER_H
#define FLEXFRINGE_CSVPARSER_H

#include "csv.hpp"
#include "input/trace.h"

class csv_header_parser {
private:
    std::unordered_map<std::string, std::set<int>> col_types;
    std::set<std::string> col_type_names;

public:
    explicit csv_header_parser(const std::vector<std::string> &headers);
    explicit csv_header_parser(const std::vector<std::string> &headers,
                               const std::vector<std::string> &col_type_names);

    const std::set<int>& get(const std::string& type) const;
};

class csv_parser {
    using ID = std::string;

private:
    std::unique_ptr<csv::CSVReader> reader;
    std::unordered_map<ID, trace*> trace_map;


public:
    csv_parser(std::unique_ptr<csv::CSVReader> reader)
    : reader(std::move(reader))
    {}

    void parse();

    void step();

};

#endif //FLEXFRINGE_CSVPARSER_H
