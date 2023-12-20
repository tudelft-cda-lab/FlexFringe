#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

class csvrow {
  private:
    const std::string delimiter;
    std::string m_line;
    std::vector<int> m_data;

  public:
    csvrow();
    explicit csvrow(const std::string& delimiter);
    std::string_view operator[](std::size_t index) const;
    [[nodiscard]] std::size_t size() const;
    void read_next_row(std::istream& str);
};

std::istream& operator>>(std::istream& str, csvrow& data);
