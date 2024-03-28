#include "sqldb.h"
#include "csv.hpp"
#include "input/abbadingoreader.h"
#include "input/inputdata.h"
#include "mem_store.h"
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

namespace psql {

db::db() : connection_string(""), table_name("benching_sample"), conn("") {
    check_connection();
    create_table(POSTGRESQL_DROPTBLS);
    create_meta_table(POSTGRESQL_DROPTBLS);
}

db::db(const std::string& table_name, const std::string& connection_string)
    : connection_string(connection_string), table_name(table_name), conn(connection_string) {
    check_connection();
    create_table(POSTGRESQL_DROPTBLS);
    create_meta_table(POSTGRESQL_DROPTBLS);
}

void db::reset() {
    conn = pqxx::connection(connection_string);
    check_connection();
}

void db::check_connection() {
    if (!conn.is_open()) {
        const auto* err = "Failed to connect to PostgreSQL.";
        std::cerr << err << std::endl;
        throw pqxx::failure(err);
    }
    LOG_S(INFO) << "Connected to PostgreSQL successfully!";
}

void db::create_table(bool drop) {
    pqxx::work tx{conn};
    if (drop)
        tx.exec0(fmt::format("DROP TABLE IF EXISTS {0};", table_name));
    tx.exec0(fmt::format(
        "CREATE TABLE IF NOT EXISTS {0} (pk serial primary key, {1} text UNIQUE NOT NULL, {2} integer NOT NULL);"
        "CREATE INDEX IF NOT EXISTS {0}_{1}_spgist ON {0} USING spgist ({1} text_ops);",
        table_name, TRACE_NAME, TYPE_NAME));
    tx.commit();
}
void db::create_meta_table(bool drop) {
    pqxx::work tx{conn};
    if (drop)
        tx.exec0(fmt::format("DROP TABLE IF EXISTS {0};", table_name + "_meta"));
    tx.exec0(fmt::format("CREATE TABLE IF NOT EXISTS {0} ({1} text UNIQUE NOT NULL, {2} text[] NOT NULL);",
                         table_name + "_meta", "name", "value"));
    tx.commit();
}

std::vector<std::string> db::get_vec_from_map(const std::map<std::string, int>& mapping) {
    std::vector<std::string> res(mapping.size());
    for (auto const map : mapping) {
        res[map.second] = map.first;
    }
    return res;
}

std::string db::get_sqlarr_from_vec(const std::vector<std::string>& vec) {
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

char db::num2str(int num) {
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

int db::str2num(char str) {
    int x = static_cast<int>(str);
    if (x > 96) {
        return x - 65 - 6;
    }
    return x - 65;
}

std::string db::vec2str(const std::vector<int>& vec) {
    std::stringstream ss;
    for (auto x : vec) {
        ss << num2str(x);
    }
    return ss.str();
}
std::vector<int> db::str2vec(const std::string& str) {
    std::vector<int> vec;
    for (char c : str) {
        vec.push_back(str2num(c));
    }
    return vec;
}

std::vector<std::string> db::get_alphabet() {
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

std::vector<std::string> db::get_types() {
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

void db::load_traces(abbadingo_inputdata& id, std::ifstream& input_stream) {
    bool check_dups = false; // if there are duplicates, this check is. Set false if low memory footprint required.
    LOG_S(INFO) << "Loading traces";

    pqxx::work tx = pqxx::work(conn);
    pqxx::stream_to stream = pqxx::stream_to::raw_table(tx, table_name, TRACE_NAME + ", " + TYPE_NAME);

    std::set<std::string> inserted;

    id.read_abbadingo_header(input_stream);
    for (int i = 0; i < id.get_max_sequences(); ++i) {
        trace* tr = mem_store::create_trace();
        id.read_abbadingo_sequence(input_stream, tr);
        auto trace = vec2str(tr->get_input_sequence());
        auto type = tr->get_type();
        if (check_dups) {
            if (inserted.contains(trace))
                continue;
        }
        inserted.insert(trace);
        stream << std::tuple(trace, type);
        tr->erase();
    }

    stream.complete();
    tx.commit();
    LOG_S(INFO) << "Loaded traces";

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
        LOG_S(INFO) << "Created meta";
    }
}

void db::copy_data(const std::string& file_name, char delimiter) {
    csv::CSVFormat format;
    format.delimiter(delimiter);
    csv::CSVReader reader(file_name, format);
    pqxx::work tx = pqxx::work(conn);
    pqxx::stream_to stream = pqxx::stream_to::raw_table(tx, table_name, TRACE_NAME + ", " + TYPE_NAME);
    for (csv::CSVRow& row : reader) {
        std::vector<std::string> data;
        for (csv::CSVField field : row) {
            data.push_back(field.get<>());
        }
        stream << std::tuple(data[0], data[1]);
    }
    stream.complete();
    tx.commit();
}

void db::prefix_query(const std::string& prefix, const std::function<bool(record)>& func) {
    pqxx::work tx{conn};
    const std::string query =
        fmt::format("SELECT pk, {0}, {1} FROM {2} WHERE {0} LIKE '{3}%'", TRACE_NAME, TYPE_NAME, table_name, prefix);
    for (auto const& [pk, trace, type] : tx.stream<int, std::string, int>(query)) {
        record r{pk, str2vec(trace), type};
        if (!func(r))
            break;
    }
    tx.commit();
}

record db::distinguish_query(const std::string& trace1, const std::string& trace2) {
    pqxx::work tx{conn};
    const std::string query =
        fmt::format("SELECT, pk, {0}, {1} FROM ("                                                              //
                    "    SELECT * FROM {2} WHERE {0} LIKE '{3}%') q1"                                          //
                    "INNER JOIN ("                                                                             //
                    "    SELECT * FROM {0} WHERE {1} LIKE  '{4}%') q2"                                         //
                    "ON substring(q1.{0} FROM (LENGTH('{3}') + 1)) = substring(q2.{0} FROM LENGTH('{4}') + 1)" //
                    "WHERE q1.{1} != q2.{1} LIMIT 1;",
                    TRACE_NAME, TYPE_NAME, table_name, trace1, trace2);
    auto [pk, trace, type] = tx.query1<int, std::string, int>(query);
    record r{pk, str2vec(trace), type};
    DLOG_S(1) << query << " answered with " << trace;
    return r;
}

std::vector<record> db::prefix_query(const std::string& prefix, int k) {
    pqxx::work tx{conn};
    const std::string query = fmt::format("SELECT pk, {0}, {1} FROM {2} WHERE {0} LIKE '{3}%' LIMIT {4}", TRACE_NAME,
                                          TYPE_NAME, table_name, prefix, k);
    std::vector<record> ans;
    for (auto const& [pk, trace, type] : tx.stream<int, std::string, int>(query)) {
        ans.push_back(record(pk, str2vec(trace), type));
    }
    DLOG_S(1) << query << " answered with " << ans.size();
    return ans;
}

void db::add_row(const std::string& trace, int type) {
    pqxx::work tx{conn};
    tx.exec0(
        fmt::format("INSERT INTO {0} ({1}, {2}) VALUES ('{3}', {4})", table_name, TRACE_NAME, TYPE_NAME, trace, type));
    tx.commit();
}

int db::query_trace(const std::string& trace) {
    auto query = fmt::format("SELECT {0} FROM {1} WHERE {2} = '{3}'", TYPE_NAME, table_name, TRACE_NAME, trace);
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

int db::query_trace_maybe(const std::string& trace) {
    auto query = fmt::format("SELECT COALESCE( (SELECT {0} FROM {1} WHERE {2} = '{3}'), -1)", TYPE_NAME, table_name,
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

std::optional<record> db::query_trace_opt(const std::string& trace) {
    auto query = fmt::format("SELECT COALESCE( (SELECT pk FROM {1} WHERE {2} = '{3}'), -1)", "", table_name, TRACE_NAME,
                             trace); // The last -1 here is the default value.
    DLOG_S(1) << query;
    try {
        pqxx::work tx{conn};
        auto val = tx.query_value<int>(query);
        tx.commit();
        LOG_S(INFO) << "Query:" << query << ":" << trace << ":" << val;
        if (val <= -1)
            return std::nullopt;
        return select_by_pk(val);
    } catch (const pqxx::unexpected_rows& e) {
        std::string exc{e.what()};
        trim(exc);
        throw std::runtime_error(exc + " Query: " + query);
    }
}

bool db::is_member(const std::string& trace) {
    pqxx::work tx{conn};
    auto val = tx.query_value<bool>(
        fmt::format("SELECT EXISTS (SELECT * FROM {0} WHERE {1} = '{2}' )", table_name, TRACE_NAME, trace));
    tx.commit();
    return val;
}

std::optional<record> db::regex_equivalence(const std::string& regex, int type) {
    auto query = fmt::format("SELECT pk, {1}, {3} FROM {0} WHERE {1} ~ '{2}' and {3} != {4} LIMIT 1", table_name,
                             TRACE_NAME, regex, TYPE_NAME, type);
    /* DLOG_S(1) << query; */
    try {
        pqxx::work tx{conn};
        auto [pk, trace, type] = tx.query1<int, std::string, int>(query);
        tx.commit();

        auto trace_vec = str2vec(trace);
        return std::make_optional<record>(record(pk, trace_vec, type));
    } catch (const pqxx::unexpected_rows& e) {
        std::string exc{e.what()};
        trim(exc);
        LOG_S(INFO) << "Found no counter example: " << exc << " Query: " << query;
        return std::nullopt;
    }
}

std::optional<record> db::select_by_pk(int pk) {
    auto query = fmt::format("SELECT {0}, {1} FROM {2} WHERE pk = {3}", TRACE_NAME, TYPE_NAME, table_name, pk);
    try {
        pqxx::work tx{conn};
        auto [trace, type] = tx.query1<std::string, int>(query);
        tx.commit();
        auto trace_vec = str2vec(trace);
        return std::make_optional<record>(record(pk, trace_vec, type));
    } catch (const pqxx::unexpected_rows& e) {
        std::string exc{e.what()};
        trim(exc);
        LOG_S(INFO) << exc << " Query: " << query;
        return std::nullopt;
    }
}

int db::max_trace_pk() {
    if (max_pk > 0)
        return max_pk;

    auto query = fmt::format("SELECT MAX(pk) FROM {0}", table_name);
    pqxx::work tx{conn};
    int val = tx.query_value<int>(query);
    tx.commit();
    return val;
}

} // namespace psql
