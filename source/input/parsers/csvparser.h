//
// Created by tom on 1/4/23.
//

#ifndef FLEXFRINGE_CSVPARSER_H
#define FLEXFRINGE_CSVPARSER_H

#include "i_parser.h"
#include "csv.hpp"
#include "input/trace.h"
#include "symbol_info.h"
#include <optional>

class csv_header_parser {
    using TypeName = std::string;
private:
    static const std::set<std::string> default_col_type_names;
    static const std::set<std::string> reserved_col_type_names;
    std::set<std::string> col_type_names;
    std::optional<std::set<std::string>> non_reserved_col_type_names {};

    std::unordered_map<TypeName, std::set<int>> col_types;
    std::unordered_map<TypeName, std::vector<std::string>> col_names;
    std::unordered_map<size_t, std::set<char>> attr_types;

    void setup_col_maps();

    void parse(const std::vector<std::string> &headers);

    void check_duplicates(const std::string &col_type) const;

public:
    explicit csv_header_parser(const std::vector<std::string> &headers);

    explicit csv_header_parser(const std::vector<std::string> &headers,
                               const std::set<std::string> &col_type_names);

    const std::set<int> &get(const std::string &type) const;

    const std::vector<std::string> &get_names(const std::string &type) const;

    const std::set<std::string> &get_column_type_names() const;

    std::set<std::string> get_non_reserved_column_type_names();
    const std::set<char> &get_col_attr_types(size_t idx) const;

};

class csv_parser : public parser {
    using ID = std::string;

private:
    std::unique_ptr<csv::CSVReader> reader;
    std::unique_ptr<csv_header_parser> header_parser;
    std::unordered_map<std::string, std::shared_ptr<std::vector<attribute_info>>> tattr_info;

public:
    explicit csv_parser(std::unique_ptr<csv::CSVReader> reader)
            : reader(std::move(reader)) {

        std::vector<std::string> col_names = this->reader->get_col_names();

        // TODO: maybe DI this?
        header_parser = std::make_unique<csv_header_parser>(col_names);
    }

    template<typename... Args>
    explicit csv_parser(Args&&... args)
    : reader(std::make_unique<csv::CSVReader>(std::forward<Args>(args)...))
    {
        std::vector<std::string> col_names = this->reader->get_col_names();
        header_parser = std::make_unique<csv_header_parser>(col_names);
    }

    std::optional<symbol_info> next() override;

    [[maybe_unused]]
    std::string get_str_from_row(const std::string &label, const csv::CSVRow &row);

    std::vector<std::string> get_vec_from_row(const std::string &label, const csv::CSVRow &row);
};

#endif //FLEXFRINGE_CSVPARSER_H
