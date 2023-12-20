#include "sqldb.h"
#include "csvrow.h"
#include "input/inputdata.h"
#include <fmt/format.h>
#include <iostream>
#include <pqxx/pqxx>
#include "loguru.hpp"

sqldb::sqldb() : connection_string(""), table_name("benching_sample"), conn("") {
  check_connection();
}

sqldb::sqldb(const std::string& table_name, const std::string& connection_string)
  : connection_string(connection_string), table_name(table_name), conn(connection_string) {
  check_connection();
}

void sqldb::reset() {
  conn = pqxx::connection(connection_string);
  check_connection();
}

void sqldb::check_connection() {
  if (!conn.is_open()) {
    const auto* err = "Failed to connect to PostgreSQL.";
    std::cerr << err << std::endl;
    throw pqxx::failure(err);
  }
  LOG_S(INFO) << "Connected to PostgreSQL successfully!" << std::endl;
}

void sqldb::create_table(bool drop = false) {
  pqxx::work tx{conn};
  if (drop) tx.exec0(fmt::format("DROP TABLE IF EXISTS {0};", table_name));
  tx.exec0(fmt::format("CREATE TABLE {0} ({1} text NOT NULL, {2} boolean NOT NULL);"
                       "CREATE INDEX {0}_{1}_spgist ON {0} USING spgist ({1} text_ops);",
                       table_name, TRACE_NAME, RES_NAME));
  tx.commit();
}

void sqldb::load_traces(inputdata& id) {
  LOG_S(INFO) << "Loading traces" << std::endl;
  // TODO
}

bool sqldb::get_state(const std::string& trace) {
  pqxx::work tx{conn};
  bool val = tx.query_value<bool>(
      fmt::format("SELECT {0} FROM {1} WHERE {2} = '{3}'", RES_NAME, table_name, TRACE_NAME, trace));
  tx.commit();
  return val;
}

void sqldb::copy_data(const std::string& file_name, const std::string& delimiter) {
  std::ifstream file(file_name);
  csvrow row(delimiter);
  pqxx::work tx = pqxx::work(conn);
  pqxx::stream_to stream = pqxx::stream_to::raw_table(tx, table_name, TRACE_NAME + ", " + RES_NAME);
  while (file >> row) { stream << std::tuple(row[0], row[1]); }
  stream.complete();
  tx.commit();
}

void sqldb::prefix_query(const std::string& prefix, const std::function<bool(std::string, bool)>& func) {
  pqxx::work tx = pqxx::work(conn);
  const std::string query =
      fmt::format("SELECT {0}, {1} FROM {2} WHERE {0} LIKE '{3}%'", TRACE_NAME, RES_NAME, table_name, prefix);
  for (auto const& [trace, res] : tx.stream<std::string, bool>(query)) {
    bool cont = func(trace, res);
    if (!cont) break;
  }
  tx.commit();
}

void sqldb::add_row(const std::string& trace, bool res) {
  pqxx::work tx{conn};
  tx.exec0(fmt::format("INSERT INTO {0} ({1}, {2}) VALUES ('{3}', {4})", table_name, TRACE_NAME, RES_NAME, trace, res));
  tx.commit();
}
