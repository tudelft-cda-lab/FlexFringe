/* https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c */

#include "csvrow.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

csvrow::csvrow() : delimiter(",") {}

csvrow::csvrow(const std::string& delimiter) : delimiter(delimiter) {}

std::string_view csvrow::operator[](std::size_t index) const {
    auto res = std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] - (m_data[index] + 1));
    return res;
}

[[nodiscard]] std::size_t csvrow::size() const { return m_data.size() - 1; }

void csvrow::read_next_row(std::istream& str) {
    std::getline(str, m_line);

    m_data.clear();
    m_data.emplace_back(-1);
    std::string::size_type pos = 0;
    while ((pos = m_line.find(delimiter, pos)) != std::string::npos) {
        m_data.emplace_back(pos);
        ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos = m_line.size();
    m_data.emplace_back(pos);
}

std::istream& operator>>(std::istream& str, csvrow& data) {
    data.read_next_row(str);
    return str;
}
