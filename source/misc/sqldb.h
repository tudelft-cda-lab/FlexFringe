/**
 * @file sqldb_sul.cpp
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief Connect to a database and make queries
 * @version 0.1
 * @date 2023-04-06
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _SQLDB_H_
#define _SQLDB_H_

#include "input/inputdata.h"
#include <memory>
#include <pqxx/pqxx>

class sqldb {
  private:
    const std::string connection_string;
    pqxx::connection conn;

    std::vector<std::string> get_vec_from_map(std::map<std::string, int> mapping);
    std::string get_sqlarr_from_vec(std::vector<std::string> vec);
    const std::string table_name;

  public:
    sqldb();
    explicit sqldb(const std::string& table_name, const std::string& connection_string = "");

    pqxx::connection& get_connection() { return conn; }

    inline static const std::string TRACE_NAME = "trace";
    inline static const std::string RES_NAME = "res";

    char num2str(int num);
    int str2num(char str);
    std::vector<int> str2vec(std::string str);
    std::string vec2str(std::vector<int> vec);

    const std::string get_table_name() { return table_name; };
    std::vector<std::string> get_alphabet();
    std::vector<std::string> get_types();
    void check_connection();
    void reset();
    void create_table(bool drop = false);
    void create_meta_table(bool drop = false);

    /**
     * @brief Load all traces from inputdata into the database.
     * Also initializing the _meta table with the alphabets and output symbols.
     */
    void load_traces(inputdata& id);
    void add_row(const std::string& trace, int res);
    void copy_data(const std::string& file_name, const std::string& delimiter = "\t");
    void tester(const std::string& val, const std::function<void(std::string)>& func);

    /**
     * @brief Loop over all queries starting with prefix. Performing function func, continuing as func returns true.
     */
    void prefix_query(const std::string& prefix, const std::function<bool(std::string, int)>& func);

    /**
     * @brief Return the type of the trace.
     * Crashes when trace not in database.
     */
    int query_trace(const std::string& trace);

    /**
     * @brief Return the type of the trace with maybe unknown type (-1).
     */
    int query_trace_maybe(const std::string& trace);

    /**
     * @brief Check if the trace can be found in the database.
     */
    bool is_member(const std::string& trace);
};

#endif
