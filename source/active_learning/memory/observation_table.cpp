/**
 * @file observation_table.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "observation_table.h"
#include "common_functions.h"
#include "definitions.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std;
using namespace obs_table_namespace;
using namespace active_learning_namespace;

/**
 * @brief Construct a new observation_table::observation_table object
 *
 * @param alphabet We need the alphabet beforehand to know how to extend, see e.g. "Learning regular sets from queries
 * and counterexamples" by Dana Angluin.
 */
observation_table::observation_table(const vector<int>& alphabet) : alphabet(alphabet), checked_for_closedness(false) {

    // initialize the lower table properly
    for (const auto i : alphabet) {
        pref_suf_t new_row_name{i};

        incomplete_rows.push_back(pref_suf_t(new_row_name)); // we do a copy to circumvent the destructor
        table_mapper[pref_suf_t(new_row_name.begin(), new_row_name.end())] =
            upper_lower_t::lower; // TODO: do we need the copy of the prefix here?
        lower_table[std::move(new_row_name)] = map<pref_suf_t, int>();
    }

    const auto nullv = get_null_vector();
    incomplete_rows.push_back(nullv); // we do a copy to circumvent the destructor
    table_mapper[pref_suf_t(nullv.begin(), nullv.end())] =
        upper_lower_t::lower; // TODO: do we need the copy of the prefix here?
    lower_table[std::move(nullv)] = map<pref_suf_t, int>();

    all_columns.insert(get_null_vector());
};

/**
 * @brief Checks if the record is in the table. We separated because we have two tables, hence avoids duplicate code.
 *
 * @param selected_table The selected table, upper or lower one.
 * @param row Row.
 * @param col Col.
 * @return true Entry exists.
 * @return false Doesn't exist.
 */
const bool observation_table::record_is_in_selected_table(const table_type& selected_table, const pref_suf_t& raw_row,
                                                          const pref_suf_t& col) const {
    const auto row = map_prefix(raw_row);
    if (!selected_table.contains(row)) {
        throw logic_error("The row should exist. What happened?");
    }
    return selected_table.at(row).contains(map_prefix(col));
}

/**
 * @brief This function exists for convenience. The empty prefix is by design the nullvector, hence we need to check we
 * got this one straight.
 *
 * @param column_name The name of the column.
 * @return const pref_suf_t The column again or the nullvector.
 */
const pref_suf_t observation_table::map_prefix(const pref_suf_t& column) const {
    if (column.size() == 0) {
        return get_null_vector();
    }
    return column;
}

/**
 * @brief Checks if record already exists in table. If it does we can save time.
 *
 * @param row Row.
 * @param col Col.
 * @return true Entry exists.
 * @return false Doesn't exist.
 */
const bool observation_table::has_record(const pref_suf_t& raw_row, const pref_suf_t& col) const {
    const auto row = map_prefix(raw_row);
    if (!table_mapper.contains(row)) {
        throw logic_error("Why do we ask about a row that does not exist? (observation_table::has_record)");
    }

    auto table_select = table_mapper.at(row);
    switch (table_select) {
    case upper_lower_t::upper:
        return record_is_in_selected_table(upper_table, row, map_prefix(col));
        break;
    case upper_lower_t::lower:
        return record_is_in_selected_table(lower_table, row, map_prefix(col));
        break;
    default:
        throw runtime_error("Unknown table_select variable occured in observation_table::has_record.");
    }
}

/**
 * @brief What you think it does.
 */
int observation_table::get_answer_from_selected_table(const table_type& selected_table, const pref_suf_t& row,
                                                      const pref_suf_t& col) const {
    return selected_table.at(map_prefix(row)).at(map_prefix(col));
}

/**
 * @brief What you think it does.
 *
 * @param raw_row The prefix.
 * @param col The suffix.
 * @return int Answer.
 */
int observation_table::get_answer(const active_learning_namespace::pref_suf_t& raw_row,
                                  const active_learning_namespace::pref_suf_t& col) const {
    const auto row = map_prefix(raw_row);
    if (!table_mapper.contains(row)) {
        throw logic_error("Why do we ask about a row that does not exist? (observation_table::get_answer)");
    }

    auto table_select = table_mapper.at(row);
    switch (table_select) {
    case upper_lower_t::upper:
        return get_answer_from_selected_table(upper_table, row, map_prefix(col));
        break;
    case upper_lower_t::lower:
        return get_answer_from_selected_table(lower_table, row, map_prefix(col));
        break;
    default:
        throw runtime_error("Unknown table_select variable occured in observation_table::get_answer.");
    }
}

/**
 * @brief Insert the record in selected (upper or lower) table. This function avoids duplicate code.
 *
 * @param selected_table Upper or lower table.
 * @param row The row.
 * @param col The column.
 * @param answer The answer to insert, as returned by the oracle.
 */
void observation_table::insert_record_in_selected_table(table_type& selected_table, const pref_suf_t& raw_row,
                                                        const pref_suf_t& col, const int answer) {
    const auto row = map_prefix(raw_row);
    if (!selected_table.contains(row)) {
        throw logic_error("The row should exist. What happened?");
    }
    selected_table[row][map_prefix(col)] = answer;
}

/**
 * @brief Inserts a record with the known answer into the table. The answer has been obtained by the teacher.
 *
 * @param raw_row The row/prefix.
 * @param col The column/suffix.
 * @param answer The answer.
 */
void observation_table::insert_record(const pref_suf_t& raw_row, const pref_suf_t& col, const int answer) {
    const auto row = map_prefix(raw_row);
    if (!table_mapper.contains(row))
        throw logic_error("Why do we ask about a row that does not exist? (observation_table::insert_record).");

    auto table_select = table_mapper.at(row);
    switch (table_select) {
    case upper_lower_t::upper:
        insert_record_in_selected_table(upper_table, row, map_prefix(col), answer);
        break;
    case upper_lower_t::lower:
        insert_record_in_selected_table(lower_table, row, map_prefix(col), answer);
        break;
    default:
        throw runtime_error("Unknown table_select variable occured in observation_table::insert_record.");
    }
}

/**
 * @brief Clears the entire set and rehashes all rows. Used after finding a counterexample and extending the columns.
 *
 */
void observation_table::hash_upper_table() {
    // upper_table_rows.clear();
    for (auto it = upper_table.cbegin(); it != upper_table.cend(); ++it) {
        const auto& entry = it->second;
        upper_table_rows.insert(entry);
    }
}

/**
 * @brief Used when we move a row to the upper table. Updates all data structures accordingly.
 *
 * @param row The row to move.
 */
void observation_table::move_to_upper_table(const active_learning_namespace::pref_suf_t& raw_row) {
    const auto row = map_prefix(raw_row);
    if (lower_table.contains(row) && upper_table.contains(row)) {
        throw logic_error("Invariant broken. The two tables should never have the same row the same time.");
    }

    const auto& entry = lower_table.at(row);

    upper_table[row] = entry;
    upper_table_rows.insert(std::move(entry));

    lower_table.erase(row);
    table_mapper.at(row) = upper_lower_t::upper;
}

/**
 * @brief This function has two purposes. It checks if the table is closed as given by the algorithm. It also moves all
 * the unique entries from the lower table to the upper table. (Perhaps we should separate those two functionalities?)
 *
 * A table is closed then all rows of the lower table also exist in the upper table. If we do have a row in the lower
 * table that does not exist in the upper table, then we identified a new state, and hence we move the row from the
 * lower table to the upper table. The entries in the upper table represent the unique states in the end.
 *
 * @return true Table is closed.
 * @return false Table not closed.
 */
const bool observation_table::is_closed() {
    if (checked_for_closedness) {
        throw logic_error(
            "is_closed() cannot be called consecutively without extending columns or lower table in the meantime.");
    }

    checked_for_closedness = true;
    hash_upper_table();

    bool is_closed = true;

    // we break the lower_table if we delete on the fly, hence we need storage
    set<pref_suf_t> rows_to_move;
    for (const auto& it : lower_table) {
        const auto& entry = it.second;
        if (!upper_table_rows.contains(entry)) {
            is_closed = false;
            rows_to_move.insert(it.first);
        }
    }

    for (const auto& row : rows_to_move) {
        move_to_upper_table(row);
    }

    return is_closed;
}

/**
 * @brief Gets the incomplete rows. Helps speeding up the algorithm by saving the search.
 *
 * @return const list< pref_suf_t >& The list of incomplete rows.
 */
const list<pref_suf_t>& observation_table::get_incomplete_rows() const { return this->incomplete_rows; }

/**
 * @brief Delete the row from the incomplete_rows data structure.
 *
 * @param row The row to close/complete.
 */
void observation_table::mark_row_complete(const pref_suf_t& row) {
    auto position_it = std::find(incomplete_rows.begin(), incomplete_rows.end(), map_prefix(row));
    incomplete_rows.erase(position_it);
}

/**
 * @brief Extends the columns by all the prefixes the argument suffix includes. Marks all rows as incomplete, as they
 * are by design again.
 *
 * NOTA BENE:
 * Hello person, Hielke here.
 * I believe this is wrongly implemented.
 * I think Robert confuses original Angluin Lstar where the prefixes are added to the RED set,
 * with modern Lstar with Shahbaz and Groz counterexample processing
 * where all the SUFFIXES are added to the experiment set.
 * So it should be
 *
 * std::deque<int> current_suffix;
 * for (const int symbol : ranges::view::reverse(suffix) {
 *     current_suffix.push_front(symbol);
 *     all_columns.emplace(current_suffix.begin(), current_suffix.end());
 * }
 *
 * for modern Lstar counter example processing.
 * But do your own research.
 *
 * @param suffix The suffix by which to extend. Gained from a counterexample as per L* algorithm.
 */
void observation_table::extent_columns(const pref_suf_t& suffix) {
    checked_for_closedness = false;

    pref_suf_t current_suffix;
    for (const int symbol : suffix) {
        current_suffix.push_back(symbol);
        all_columns.insert(current_suffix);
    }

    incomplete_rows.clear();
    for (auto it = upper_table.cbegin(); it != upper_table.cend(); ++it) {
        const auto& row_name = it->first;
        incomplete_rows.push_back(row_name);
    }
    for (auto it = lower_table.cbegin(); it != lower_table.cend(); ++it) {
        const auto& row_name = it->first;
        incomplete_rows.push_back(row_name);
    }
}

/**
 * @brief Extends the lower table by adding an extra character to each table. These new rows are by design empty and
 * hence incomplete_rows is updated.
 *
 */
void observation_table::extend_lower_table() {
    checked_for_closedness = false;

    // adding to lower table while iterating its results in infinite loop, hence we do auxiliary object
    set<pref_suf_t> all_row_names;
    for (auto it = lower_table.cbegin(); it != lower_table.cend(); ++it) {
        const auto& row_name = it->first;
        all_row_names.insert(row_name);
    }

    for (auto it = upper_table.cbegin(); it != upper_table.cend(); ++it) {
        const auto& row_name = it->first;
        all_row_names.insert(row_name);
    }

    for (const auto& row_name : all_row_names) {
        for (const auto i : alphabet) {
            pref_suf_t new_row_name = pref_suf_t(row_name.begin(), row_name.end());
            new_row_name.push_back(i);

            if (lower_table.contains(new_row_name) || upper_table.contains(new_row_name))
                continue;

            incomplete_rows.push_back(new_row_name);           // we do a copy to circumvent the destructor
            table_mapper[new_row_name] = upper_lower_t::lower; // TODO: do we need the copy of the prefix here?
            lower_table[move(new_row_name)] = row_type();
        }
    }
}

/**
 * @brief For debugging purposes. Prints all the rows and columns.
 *
 */
void observation_table::print() const {
    /*   cout << "Upper table: " << endl;
      for(auto it = upper_table.cbegin(); it != upper_table.cend(); ++it){
        const auto& row_name = it->first;
        print_vector(row_name);
      }

      cout << "Lower table:" << endl;
      for(auto it = lower_table.cbegin(); it != lower_table.cend(); ++it){
        const auto& row_name = it->first;
        print_sequence(row_name);
      }

      cout << "Columns:" << endl;
      for(const auto col: all_columns){
        print_vector(col);
      }

      cout << "Rows to close:" << endl;
      for(const auto r: incomplete_rows){
        print_vector(r);
      } */
}
