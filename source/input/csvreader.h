#ifndef FLEXFRINGE_CSVREADER_H
#define FLEXFRINGE_CSVREADER_H

#include <istream>
#include <string>
#include <set>
#include <map>
#include <unordered_map>
#include <list>

#include "input/inputdata.h"
#include "input/trace.h"
#include "input/tail.h"
#include "input/attribute.h"
#include "input/inputdatalocator.h"




class csv_inputdata: public inputdata {
private:
    std::unordered_map<std::string, trace*> trace_map;

    std::set<int> id_cols;
    std::set<int> type_cols;
    std::set<int> symbol_cols;
    std::set<int> data_cols;
    std::set<int> trace_attr_cols;
    std::set<int> symbol_attr_cols;

    char delim = ',';
    bool strip_whitespace = true;

    tail *make_tail(const std::string &id,
                    const std::string &symbol,
                    const std::string &type,
                    const std::vector<std::string> &trace_attrs,
                    const std::vector<std::string> &symbol_attrs,
                    const std::vector<std::string> &data);

    void add_type_to_trace(trace *new_trace,
                           const std::string &type,
                           const std::vector<std::string> &trace_attrs);

public:


    void read(std::istream &input_stream) override;

    trace* readRow(std::istream &input_stream);
    void readHeader(std::istream &input_stream);

    // Config options
    csv_inputdata& setDelimiter(char d) {
        delim = d;
        return *this;
    }
    csv_inputdata& stripWhitespace(bool strip) {
        strip_whitespace = strip;
        return *this;
    }

    // Const getters for private data, for testing etc.
    const std::set<int> &getIdCols() const;
    const std::set<int> &getTypeCols() const;
    const std::set<int> &getSymbolCols() const;
    const std::set<int> &getDataCols() const;
    const std::set<int> &getTraceAttrCols() const;
    const std::set<int> &getSymbolAttrCols() const;

    const std::vector<attribute> &getTraceAttributes() const;
    const std::vector<attribute> &getSymbolAttributes() const;

    std::string string_from_symbol(int symbol);


};

#endif //FLEXFRINGE_CSVREADER_H
