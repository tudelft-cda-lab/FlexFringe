#include "sqldb.h"
#include "csv.hpp"
#include "input/inputdata.h"
#include "misc/printutil.h"
#include "misc/trim.h"
#include "parameters.h"
#include "utility/loguru.hpp"
#include <fmt/format.h>
#include <iostream>
#include <optional>
#include <pqxx/pqxx>
#include <sstream>
#include <utility>

sqldb::sqldb() : connection_string(""), table_name("benching_sample"), conn("") {
    check_connection();
    create_table(POSTGRESQL_DROPTBLS);
    create_meta_table(POSTGRESQL_DROPTBLS);
}

sqldb::sqldb(const std::string& table_name, const std::string& connection_string)
    : connection_string(connection_string), table_name(table_name), conn(connection_string) {
    check_connection();
    create_table(POSTGRESQL_DROPTBLS);
    create_meta_table(POSTGRESQL_DROPTBLS);
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
    LOG_S(INFO) << "Connected to PostgreSQL successfully!";
}

void sqldb::create_table(bool drop) {
    pqxx::work tx{conn};
    if (drop)
        tx.exec0(fmt::format("DROP TABLE IF EXISTS {0};", table_name));
    tx.exec0(fmt::format("CREATE TABLE IF NOT EXISTS {0} ({1} text UNIQUE NOT NULL, {2} integer NOT NULL);"
                         "CREATE INDEX IF NOT EXISTS {0}_{1}_spgist ON {0} USING spgist ({1} text_ops);",
                         table_name, TRACE_NAME, RES_NAME));
    tx.commit();
}
void sqldb::create_meta_table(bool drop) {
    pqxx::work tx{conn};
    if (drop)
        tx.exec0(fmt::format("DROP TABLE IF EXISTS {0};", table_name + "_meta"));
    tx.exec0(fmt::format("CREATE TABLE IF NOT EXISTS {0} ({1} text UNIQUE NOT NULL, {2} text[] NOT NULL);",
                         table_name + "_meta", "name", "value"));
    tx.commit();
}

std::vector<std::string> sqldb::get_vec_from_map(const std::map<std::string, int>& mapping) {
    std::vector<std::string> res(mapping.size());
    for (auto const map : mapping) { res[map.second] = map.first; }
    return res;
}

std::string sqldb::get_sqlarr_from_vec(const std::vector<std::string>& vec) {
    std::stringstream ss;
    ss << "ARRAY['";
    for (size_t i = 0; i < vec.size(); i++) {
        if (i != 0) {
            ss << "', '";
        }
        ss << vec[i];
    }
    ss << "']";
    return ss.str();
}

char sqldb::num2str(int num) {
    // TODO This only supports to ASCII, increase with UTF8 (PostgreSQL should support that).
    // See https://stackoverflow.com/questions/26074090/iterating-through-a-utf-8-string-in-c11
    if (num > 51) {
        throw std::runtime_error("Got symbol bigger than 51. Only ASCII conversion possible, extend with UTF8.");
    }
    if (num > 25) {
        return static_cast<char>(num + 65 + 6); // small letters
    } else {
        return static_cast<char>(num + 65); // capital letters
    }
}

int sqldb::str2num(char str) {
    int x = static_cast<int>(str);
    if (x > 96) {
        return x - 65 - 6;
    }
    return x - 65;
}

std::string sqldb::vec2str(const std::vector<int>& vec) {
    std::stringstream ss;
    for (auto x : vec) { ss << num2str(x); }
    return ss.str();
}
std::vector<int> sqldb::str2vec(const std::string& str) {
    std::vector<int> vec;
    for (char c : str) { vec.push_back(str2num(c)); }
    return vec;
}

std::vector<std::string> sqldb::get_alphabet() {
    LOG_S(INFO) << "Getting alphabet.";
    pqxx::work tx = pqxx::work(conn);
    auto data =
        tx.query_value<std::string>(fmt::format("SELECT value from {0} where name = 'alphabet'", table_name + "_meta"));
    tx.commit();
    pqxx::array<std::string> arr{data, conn};
    std::vector<std::string> val{arr.cbegin(), arr.cend()};
    LOG_S(INFO) << val;

    return val;
}

std::vector<std::string> sqldb::get_types() {
    LOG_S(INFO) << "Getting types.";
    pqxx::work tx = pqxx::work(conn);
    auto data =
        tx.query_value<std::string>(fmt::format("SELECT value from {0} where name = 'types'", table_name + "_meta"));
    tx.commit();
    pqxx::array<std::string> arr{data, conn};
    std::vector<std::string> val{arr.cbegin(), arr.cend()};
    LOG_S(INFO) << val;

    return val;
}

void sqldb::load_traces(inputdata& id) {
    LOG_S(INFO) << "Loading traces";

    if (POSTGRESQL_DROPTBLS) {
        // TODO: Adding more data incrementally without dropping the tables might yield trouble as
        // inputdata will have a different conversion.
        // Instead a new routine has to be added that initializes the inputdata object with the alphabet and types from
        // the _meta table.
        pqxx::work txm = pqxx::work(conn);

        // Meta data is inserted in how the conversion for inputdata works.
        // This means that the position of the symbol in the alphabet is the integer value of that symbol for inputdata.
        txm.exec0(fmt::format("INSERT INTO {0} ({1}, {2}) VALUES ('{3}', {4})", table_name + "_meta", "name", "value",
                              "alphabet", get_sqlarr_from_vec(get_vec_from_map(id.get_r_alphabet()))));
        txm.exec0(fmt::format("INSERT INTO {0} ({1}, {2}) VALUES ('{3}', {4})", table_name + "_meta", "name", "value",
                              "types", get_sqlarr_from_vec(get_vec_from_map(id.get_r_types()))));
        txm.commit();
    }

    pqxx::work tx = pqxx::work(conn);
    pqxx::stream_to stream = pqxx::stream_to::raw_table(tx, table_name, TRACE_NAME + ", " + RES_NAME);

    std::set<std::string> inserted;

    for (auto* tr : id) {
        auto trace = vec2str(tr->get_input_sequence());
        auto res = tr->get_type();
        if (inserted.contains(trace))
            continue;
        inserted.insert(trace);
        stream << std::tuple(trace, res);
    }

    stream.complete();
    tx.commit();
    LOG_S(INFO) << "Loaded traces";
}

void sqldb::copy_data(const std::string& file_name, char delimiter) {
    csv::CSVFormat format;
    format.delimiter(delimiter);
    csv::CSVReader reader(file_name, format);
    pqxx::work tx = pqxx::work(conn);
    pqxx::stream_to stream = pqxx::stream_to::raw_table(tx, table_name, TRACE_NAME + ", " + RES_NAME);
    for (csv::CSVRow& row : reader) {
        std::vector<std::string> data;
        for (csv::CSVField field : row) { data.push_back(field.get<>()); }
        stream << std::tuple(data[0], data[1]);
    }
    stream.complete();
    tx.commit();
}

void sqldb::prefix_query(const std::string& prefix, const std::function<bool(std::string, int)>& func) {
    pqxx::work tx = pqxx::work(conn);
    const std::string query =
        fmt::format("SELECT {0}, {1} FROM {2} WHERE {0} LIKE '{3}%'", TRACE_NAME, RES_NAME, table_name, prefix);
    for (auto const& [trace, res] : tx.stream<std::string, int>(query)) {
        bool cont = func(trace, res);
        if (!cont)
            break;
    }
    tx.commit();
}

std::vector<std::pair<std::vector<int>, int>> sqldb::prefix_query(const std::string& prefix, int n) {
    pqxx::work tx = pqxx::work(conn);
    const std::string query = fmt::format("SELECT {0}, {1} FROM {2} WHERE {0} LIKE '{3}%' LIMIT {4}", TRACE_NAME,
                                          RES_NAME, table_name, prefix, n);
    std::vector<std::pair<std::vector<int>, int>> ans;
    for (auto const& [trace, res] : tx.stream<std::string, int>(query)) {
        ans.push_back(std::make_pair(str2vec(trace), res));
    }
    DLOG_S(1) << query << " answered with " << ans.size();
    return ans;
}

void sqldb::add_row(const std::string& trace, int res) {
    pqxx::work tx{conn};
    tx.exec0(
        fmt::format("INSERT INTO {0} ({1}, {2}) VALUES ('{3}', {4})", table_name, TRACE_NAME, RES_NAME, trace, res));
    tx.commit();
}

int sqldb::query_trace(const std::string& trace) {
    auto query = fmt::format("SELECT {0} FROM {1} WHERE {2} = '{3}'", RES_NAME, table_name, TRACE_NAME, trace);
    DLOG_S(1) << query;
    try {
        pqxx::work tx{conn};
        auto val = tx.query_value<int>(query);
        tx.commit();
        return val;
    } catch (const pqxx::unexpected_rows& e) {
        std::string exc{e.what()};
        trim(exc);
        throw std::runtime_error(exc + " Query: " + query);
    }
}

int sqldb::query_trace_maybe(const std::string& trace) {
    auto query = fmt::format("SELECT COALESCE( (SELECT {0} FROM {1} WHERE {2} = '{3}'), -1)", RES_NAME, table_name,
                             TRACE_NAME, trace); // The last -1 here is the default value.
    DLOG_S(1) << query;
    LOG_S(INFO) << "Query: " << trace;
    try {
        pqxx::work tx{conn};
        auto val = tx.query_value<int>(query);
        tx.commit();
        return val;
    } catch (const pqxx::unexpected_rows& e) {
        std::string exc{e.what()};
        trim(exc);
        throw std::runtime_error(exc + " Query: " + query);
    }
}

bool sqldb::is_member(const std::string& trace) {
    pqxx::work tx{conn};
    auto val = tx.query_value<bool>(
        fmt::format("SELECT EXISTS (SELECT * FROM {0} WHERE {1} = '{2}' )", table_name, TRACE_NAME, trace));
    tx.commit();
    return val;
}

std::optional<std::pair<std::vector<int>, int>> sqldb::regex_equivalence(const std::string& regex, int type) {
    auto query = fmt::format("SELECT {1}, {3} FROM {0} WHERE {1} ~ '{2}' and {3} != {4} LIMIT 1", table_name,
                             TRACE_NAME, regex, RES_NAME, type);
    /* DLOG_S(1) << query; */
    try {
        pqxx::work tx{conn};
        auto [trace, type] = tx.query1<std::string, int>(query);
        tx.commit();

        auto trace_vec = str2vec(trace);
        return std::make_optional<std::pair<std::vector<int>, int>>(std::make_pair(trace_vec, type));
    } catch (const pqxx::unexpected_rows& e) {
        std::string exc{e.what()};
        trim(exc);
        LOG_S(INFO) << "Found no counter example: " << exc << " Query: " << query;
        return std::nullopt;
    }
}
