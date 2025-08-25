/**
 * @file misc/sqldb.h
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

#include "input/abbadingoreader.h"
#include "input/inputdata.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#ifdef __FLEXFRINGE_DATABASE
#include <pqxx/pqxx>
#endif /* __FLEXFRINGE_DATABASE */

// Short for postgresql
namespace psql {

struct record {
    int pk;
    std::vector<int> trace;
    int type;

    record(int i, std::vector<int> t, int o) : trace(std::move(t)), type(o), pk(i) {}
};

/* Compile a dummy psql::db when disabled to allow compiling on platforms without psql and pqxx */
#ifdef __FLEXFRINGE_DATABASE
class db {
  private:
    pqxx::connection conn;
    pqxx::connection& get_connection() { return conn; }
#else
class db {
  private:
#endif /* __FLEXFRINGE_DATABASE */

    const std::string connection_string;
    std::vector<std::string> get_vec_from_map(const std::unordered_map<std::string, int>& mapping);
    std::string get_sqlarr_from_vec(const std::vector<std::string>& vec);
    const std::string table_name;

    int max_pk = -1;

  public:
    db();
    explicit db(const std::string& table_name, const std::string& connection_string = "");

    std::string TRACE_NAME = "trace";
    std::string TYPE_NAME = "type";

    static char num2str(int num);
    static int str2num(char str);
    static std::vector<int> str2vec(const std::string& str);
    static std::string vec2str(const std::vector<int>& vec);

    const std::string& get_table_name() { return table_name; };

    /**
     * @brief Returning the alphabet stored in the '_meta' table.
     */
    std::vector<std::string> get_alphabet();

    /**
     * @brief Returning the types stored in the '_meta' table.
     */
    std::vector<std::string> get_types();

    void check_connection();
    void reset();
    void create_table(bool drop = false);
    void create_meta_table(bool drop = false);

    /**
     * @brief Load all traces from inputdata into the database.
     * Also initializing the '_meta' table with the alphabets and output symbols.
     */
    void load_traces(abbadingo_inputdata& idb, std::ifstream& input_stream);
    void add_row(const std::string& trace, int type);
    void copy_data(const std::string& file_name, char delimiter = '\t');
    void tester(const std::string& val, const std::function<void(std::string)>& func);
    int max_trace_pk();
    std::optional<record> select_by_pk(int pk);

    /**
     * @brief Loop over all queries starting with prefix. Performing function func, continuing as func returns true.
     */
    void prefix_query(const std::string& prefix, const std::function<bool(record)>& func);
    std::vector<record> prefix_query(const std::string& prefix, int n);
    void stream_traces(const std::function<bool(record)>& func);

    record distinguish_query(const std::string& trace1, const std::string& trace2);

    /**
     * @brief Return the type of the trace.
     * Crashes when trace not in database.
     */
    int query_trace(const std::string& trace);

    /**
     * @brief Return the type of the trace with maybe unknown type (-1).
     */
    int query_trace_maybe(const std::string& trace);
    std::optional<record> query_trace_opt(const std::string& trace);

    /**
     * @brief Check if the trace can be found in the database.
     */
    bool is_member(const std::string& trace);

    /**
     * @brief Looks in the database for a regex match for the specific type that is wrong.
     *
     * This function can be used to check for equivalence by checking all the types and
     * providing a regex that should match all traces outputting that specific type.
     * (You can use regex_builder to create such regexes.)
     */
    std::optional<record> regex_equivalence(const std::string& regex, int type);
};

} // namespace psql

#endif /* _SQLDB_H_ */
