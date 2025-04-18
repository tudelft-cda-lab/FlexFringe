/**
 * @file observation_table_imat.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "observation_table_imat.h"
#include "base_oracle.h"
#include "common_functions.h"
#include "definitions.h"
#include "inputdata.h"
#include "misc/utils.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std;
using namespace obs_table_imat_namespace;
using namespace active_learning_namespace;

/**
 * @brief Construct a new observation_table_imat::observation_table object
 *
 * @param alphabet We need the alphabet beforehand to know how to extend, see e.g. "Learning regular sets from queries
 * and counterexamples" by Dana Angluin.
 */
observation_table_imat::observation_table_imat(const vector<int>& alphabet)
    : alphabet(alphabet), checked_for_closedness(false) {

    // initialize the lower table properly
    for (const auto i : alphabet) {
        pref_suf_t new_row_name{i};

        incomplete_rows.push_back(pref_suf_t(new_row_name)); // we do a copy to circumvent the destructor
        table_mapper[pref_suf_t(new_row_name.begin(), new_row_name.end())] =
            upper_lower_t::lower; // TODO: do we need the copy of the prefix here?
        lower_table[std::move(new_row_name)] = {std::numeric_limits<int>::max()};
    }

    const auto nullv = get_null_vector();
    incomplete_rows.push_back(nullv); // we do a copy to circumvent the destructor
    table_mapper[pref_suf_t(nullv.begin(), nullv.end())] =
        upper_lower_t::upper; // TODO: do we need the copy of the prefix here?
    upper_table[std::move(nullv)] = {std::numeric_limits<int>::max()};

    all_columns.insert(nullv);
    exp2ind[nullv] = 0;
    new_ind++;
};

/**
 * @brief This function exists for convenience. The empty prefix is by design the nullvector, hence we need to check we
 * got this one straight.
 *
 * @param column_name The name of the column.
 * @return const pref_suf_t The column again or the nullvector.
 */
const pref_suf_t observation_table_imat::map_prefix(const pref_suf_t& column) const {
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
const bool observation_table_imat::has_record(const pref_suf_t& raw_row, const pref_suf_t& col) {
    auto row = map_prefix(raw_row);
    if (!table_mapper.contains(row)) {
        throw logic_error("Why do we ask about a row that does not exist? (observation_table_imat::has_record)");
    }

    auto table_select = table_mapper.at(row);
    row_type selected_row;
    switch (table_select) {
    case upper_lower_t::upper:
        selected_row = upper_table.at(row);
        break;
    case upper_lower_t::lower:
        selected_row = lower_table.at(row);
        break;
    default:
        throw runtime_error("Unknown table_select variable occured in observation_table::has_record.");
    }
    int ind = exp2ind.at(col);
    if (selected_row.size() <= ind)
        return false;
    if (selected_row.at(ind) == std::numeric_limits<int>::max())
        return false;
    return true;
}

/**
 * @brief What you think it does.
 */
int observation_table_imat::get_answer_from_selected_table(const table_type& selected_table, const pref_suf_t& row,
                                                           const pref_suf_t& col) const {
    auto new_col = map_prefix(col);
    int ind = exp2ind.at(new_col);
    return selected_table.at(map_prefix(row))[ind];
}

/**
 * @brief What you think it does.
 *
 * @param raw_row The prefix.
 * @param col The suffix.
 * @return int Answer.
 */
int observation_table_imat::get_answer(const active_learning_namespace::pref_suf_t& raw_row,
                                       const active_learning_namespace::pref_suf_t& col) const {
    const auto row = map_prefix(raw_row);
    if (!table_mapper.contains(row)) {
        throw logic_error("Why do we ask about a row that does not exist? (observation_table_imat::get_answer)");
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
        throw runtime_error("Unknown table_select variable occured in observation_table_imat::get_answer.");
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
void observation_table_imat::insert_record_in_selected_table(table_type& selected_table, const pref_suf_t& raw_row,
                                                             const pref_suf_t& col, const int answer) {
    const auto row = map_prefix(raw_row);
    if (!selected_table.contains(row)) {
        throw logic_error("The row should exist. What happened?");
    }

    vector<int> dest;
    dest.insert(dest.end(), row.begin(), row.end());
    dest.insert(dest.end(), col.begin(), col.end());
    DLOG_S(INFO) << row << col << endl;
    DLOG_S(INFO) << "insert:" << dest << ":" << answer << endl;

    int pos = exp2ind[map_prefix(col)];
    selected_table[row].resize(new_ind, std::numeric_limits<int>::max());
    selected_table[row][pos] = answer;
}

/**
 * @brief Inserts a record with the known answer into the table. The answer has been obtained by the oracle.
 *
 * @param raw_row The row/prefix.
 * @param col The column/suffix.
 * @param answer The answer.
 */
void observation_table_imat::insert_record(const pref_suf_t& raw_row, const pref_suf_t& col, const int answer) {
    const auto row = map_prefix(raw_row);
    if (!table_mapper.contains(row))
        throw logic_error("Why do we ask about a row that does not exist? (observation_table_imat::insert_record).");

    auto table_select = table_mapper.at(row);
    switch (table_select) {
    case upper_lower_t::upper:
        insert_record_in_selected_table(upper_table, row, map_prefix(col), answer);
        break;
    case upper_lower_t::lower:
        insert_record_in_selected_table(lower_table, row, map_prefix(col), answer);
        break;
    default:
        throw runtime_error("Unknown table_select variable occured in observation_table_imat::insert_record.");
    }
}

/**
 * @brief Used when we move a row to the upper table. Updates all data structures accordingly.
 *
 * @param row The row to move.
 */
void observation_table_imat::move_to_upper_table(const active_learning_namespace::pref_suf_t& raw_row) {
    const auto row = map_prefix(raw_row);
    if (lower_table.contains(row) && upper_table.contains(row)) {
        throw logic_error("Invariant broken. The two tables should never have the same row the same time.");
    }

    const auto& entry = lower_table.at(row);

    upper_table[row] = entry;
    lower_table.erase(row);
    table_mapper.at(row) = upper_lower_t::upper;
}

/**
 * @brief This function has two purposes. It checks if the table is closed as given by the algorithm. It also moves
 * all the unique entries from the lower table to the upper table. (Perhaps we should separate those two
 * functionalities?)
 *
 * A table is closed then all rows of the lower table also exist in the upper table. If we do have a row in the
 * lower table that does not exist in the upper table, then we identified a new state, and hence we move the row
 * from the lower table to the upper table. The entries in the upper table represent the unique states in the end.
 *
 * @return true Table is closed.
 * @return false Table not closed.
 */
const bool observation_table_imat::is_closed() {
    DLOG_S(INFO) << "IS_CLOSED" << endl;
    if (checked_for_closedness) {
        throw logic_error(
            "is_closed() cannot be called consecutively without extending columns or lower table in the meantime.");
    }

    checked_for_closedness = true;
    bool is_closed = true;

    // we break the lower_table if we delete on the fly, hence we need storage
    set<pref_suf_t> rows_to_move;

    // Check every lower table row
    // if it is different from all upper table rows,
    // then it should be promoted.
    for (const auto& [pref_lower, exp_lower] : lower_table) {
        // Check for this lower table row:
        bool equal_to_one = false;

        // Compare with every upper table row:
        for (const auto& [pref_upper, exp_upper] : upper_table) {

            bool is_different = false;
            for (int ind = 0; ind < new_ind; ind++) {
                int val_lower = exp_lower[ind];
                // Ignoring missing values for comparison in IMAT
                if (val_lower == -1)
                    continue;
                int val_upper = exp_upper[ind];
                if (val_upper == -1)
                    continue;
                if (val_lower != val_upper) {
                    // Found a legit difference.
                    // Stop checking other values.
                    is_different = true;
                    break;
                }
            }
            if (!is_different) {
                // This one was in fact equal, stop checking with the other rows now.
                equal_to_one = true;
                break;
            }
        }
        if (!equal_to_one) {
            // different from all inspected rows -> promote the row to upper table.
            is_closed = false;
            rows_to_move.insert(pref_lower);
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
const list<pref_suf_t>& observation_table_imat::get_incomplete_rows() const { return this->incomplete_rows; }

/**
 * @brief Delete the row from the incomplete_rows data structure.
 *
 * @param row The row to close/complete.
 */
void observation_table_imat::mark_row_complete(const pref_suf_t& row) {
    auto position_it = std::find(incomplete_rows.begin(), incomplete_rows.end(), map_prefix(row));
    incomplete_rows.erase(position_it);
}

void observation_table_imat::complete_rows(const unique_ptr<base_oracle>& oracle, inputdata& id) {
    DLOG_S(INFO) << "COMPLETE" << endl;
    const auto& rows_to_close =
        list<pref_suf_t>(get_incomplete_rows()); // need a copy, since we're modifying structure in mark_row_complete().
    const auto& column_names = get_column_names();

    for (const auto& current_row : rows_to_close) {
        for (const auto& current_column : column_names) {
            if (has_record(current_row, current_column))
                continue;

            const int answer = oracle->ask_sul(current_row, current_column, id).GET_INT();
            insert_record(current_row, current_column, answer);
        }
        mark_row_complete(current_row);
    }
}

/**
 * @brief Extends the columns by all the prefixes the argument suffix includes. Marks all rows as incomplete, as
 * they are by design again.
 *
 * @param suffix The suffix by which to extend. Gained from a counterexample as per L* algorithm.
 */
void observation_table_imat::extent_columns(const pref_suf_t& suffix) {
    DLOG_S(INFO) << "EXTEND" << endl;
    checked_for_closedness = false;

    std::deque<int> current_suffix;
    for (const int symbol : utils::reverse(suffix)) {
        current_suffix.push_front(symbol);
        pref_suf_t suf(current_suffix.begin(), current_suffix.end());
        all_columns.insert(suf);
        exp2ind[suf] = new_ind++;
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
void observation_table_imat::extend_lower_table() {
    checked_for_closedness = false;

    // adding to lower table while iterating its results in infinite loop, hence we do auxiliary object
    set<pref_suf_t> all_row_names;

    /* for (auto it = lower_table.cbegin(); it != lower_table.cend(); ++it) { */
    /*     const auto& row_name = it->first; */
    /*     all_row_names.insert(row_name); */
    /* } */

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
            lower_table[std::move(new_row_name)] = row_type(new_ind, std::numeric_limits<int>::max());
        }
    }
}

/**
 * @brief For debugging purposes. Prints all the rows and columns.
 *
 */
void observation_table_imat::print() const {
    cout << "Upper table rows: " << endl;
    for (auto it = upper_table.cbegin(); it != upper_table.cend(); ++it) {
        const auto& row_name = it->first;
        print_vector(row_name);
    }

    cout << "Upper table data: " << endl;
    for (auto it = upper_table.cbegin(); it != upper_table.cend(); ++it) {
        const auto& row_data = it->second;
        for (const auto col : all_columns) {
            cout << row_data.at(exp2ind.at(col)) << " ";
        }
        cout << std::endl;
    }

    cout << "Lower table rows:" << endl;
    for (auto it = lower_table.cbegin(); it != lower_table.cend(); ++it) {
        const auto& row_name = it->first;
        active_learning_namespace::print_sequence(row_name.begin(), row_name.end());
    }

    cout << "Columns:" << endl;
    for (const auto col : all_columns) {
        print_vector(col);
    }

    cout << "Rows to close:" << endl;
    for (const auto r : incomplete_rows) {
        print_vector(r);
    }
}
