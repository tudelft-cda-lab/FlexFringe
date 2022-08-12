#ifndef FLEXFRINGE_CSVREADER_H
#define FLEXFRINGE_CSVREADER_H

#include <istream>
#include <string>
#include <set>
#include <map>
#include <unordered_map>
#include <list>

#include "input/i_inputdata.h"
#include "input/trace.h"
#include "input/tail.h"
#include "input/attribute.h"
#include "input/inputdatalocator.h"

//#include "inputdata.h"


class CSVInputData: public IInputData {
private:
    std::unordered_map<std::string, Trace*> trace_map;

    std::set<int> id_cols;
    std::set<int> type_cols;
    std::set<int> symbol_cols;
    std::set<int> data_cols;
    std::set<int> trace_attr_cols;
    std::set<int> symbol_attr_cols;

    // TODO: refactor so these aren't needed anymore
    void read_abbadingo_symbol(std::istream &input_stream, Tail *new_tail);
    void read_abbadingo_type(std::istream &input_stream, Trace *new_trace);
    std::string string_from_symbol(int symbol);

    char delim = ',';
    bool strip_whitespace = true;

public:


    void read(std::istream &input_stream) override;

    Trace* readRow(std::istream &input_stream);
    void readHeader(std::istream &input_stream);

    // Config options
    CSVInputData& setDelimiter(char d) {
        delim = d;
        return *this;
    }
    CSVInputData& stripWhitespace(bool strip) {
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

    const std::vector<Attribute> &getTraceAttributes() const;
    const std::vector<Attribute> &getSymbolAttributes() const;



};

#endif //FLEXFRINGE_CSVREADER_H
