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

#include <memory>
#include <pqxx/pqxx>
#include "input/inputdata.h"

class sqldb {
private:
  const std::string connection_string;
  pqxx::connection conn;

public:
  sqldb();
  explicit sqldb(const std::string& table_name, const std::string& connection_string = "");

  pqxx::connection& get_connection() { return conn; }

  inline static const std::string TRACE_NAME = "trace";
  inline static const std::string RES_NAME = "res";
  const std::string table_name;

  void check_connection();
  void reset();
  void create_table(bool drop);
  void load_traces(inputdata& id);
  void add_row(const std::string& trace, bool res);
  void copy_data(const std::string& file_name, const std::string& delimiter = "\t");
  bool get_state(const std::string& trace);
  void tester(const std::string& val, const std::function<void(std::string)>& func);
  void prefix_query(const std::string& prefix, const std::function<bool(std::string, bool)>& func);
};

#endif
