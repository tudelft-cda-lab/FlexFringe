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
#include "definitions.h"

#include <utility>
#include <stdexcept>
#include <cassert>

using namespace std;
using namespace obs_table_namespace;
using namespace active_learning_namespace;

observation_table::observation_table(vector<int>& alphabet) : alphabet(alphabet) {
  const auto nullvector = get_null_vector();
  upper_table[nullvector][nullvector] = knowledge_t::accepting;

  table_mapper[nullvector] = upper_lower_t::upper;
  all_colums.insert(nullvector);
  hash_upper_table();

  extend_lower_table();
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
const bool observation_table::record_is_in_selected_table(const table_type& selected_table, const pref_suf_t& row, const pref_suf_t& col) const {
  if(!selected_table.contains(row)){ throw logic_error("The row should exist. What happened?"); }
  //assert(selected_table.contains(row), "The row should exist. What happened?");

  return selected_table.at(row).contains(col);
}

/**
 * @brief Checks if record already exists in table. If it does we can save time.
 * 
 * @param row Row.
 * @param col Col.
 * @return true Entry exists.
 * @return false Doesn't exist.
 */
const bool observation_table::has_record(const pref_suf_t& row, const pref_suf_t& col) const {
  if(!table_mapper.contains(row)){ throw logic_error("This should not happen. Why do we ask about a row that does not exist?"); }

  auto table_select = table_mapper.at(row);
  switch(table_select){
    case upper_lower_t::upper:
      return record_is_in_selected_table(upper_table, row, col);
    case upper_lower_t::lower:
      return record_is_in_selected_table(lower_table, row, col);
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
void observation_table::insert_record_in_selected_table(table_type& selected_table, const pref_suf_t& row, const pref_suf_t& col, const knowledge_t answer){
  if(!selected_table.contains(row)){ throw logic_error("The row should exist. What happened?"); }

  selected_table[row][col] = answer;
}

void observation_table::insert_record(const pref_suf_t& row, const pref_suf_t& col, const knowledge_t answer){
  if(!table_mapper.contains(row)){ throw logic_error("This should not happen. Why do we ask about a row that does not exist?"); }
  //assert(answer != knowledge_t::unknown);

  auto table_select = table_mapper.at(row);
  switch(table_select){
    case upper_lower_t::upper:
      insert_record_in_selected_table(upper_table, row, col, answer);
    case upper_lower_t::lower:
      insert_record_in_selected_table(lower_table, row, col, answer);
    default:
      throw runtime_error("Unknown table_select variable occured in observation_table::get_answer.");
  }
}

/**
 * @brief Clears the entire set and rehashes all rows. Used after finding a counterexample and extending the columns.
 * 
 */
void observation_table::hash_upper_table(){
  upper_table_rows.clear();
  for(auto it = upper_table.cbegin(); it != upper_table.cend(); ++it){
    const auto& entry = it->second;
    upper_table_rows.insert(entry);
  }
}

/**
 * @brief Used when we move a row to the upper table. Updates all data structures accordingly.
 * 
 * @param row 
 */
void observation_table::move_to_upper_table(const active_learning_namespace::pref_suf_t& row){
  if(lower_table.contains(row) && upper_table.contains(row)){ throw logic_error("Invariant broken. The two tables should never have the same row the same time."); }

  const auto& entry = lower_table.at(row);
  upper_table[row] = entry;
  upper_table_rows.insert(entry);

  //const auto it = lower_table.find(row);
  //lower_table.erase(it);
  lower_table.erase(row);
}

/**
 * @brief This function has two purposes. It checks if the table is closed as given by the algorithm. It also moves all the 
 * unique entries from the lower table to the upper table.
 * 
 * A table is closed then all rows of the lower table also exist in the upper table. If we do have a row in the lower table 
 * that does not exist in the upper table, then we identified a new state, and hence we move the row from the lower table 
 * to the upper table. The entries in the upper table represent the unique states in the end.
 * 
 * @return true Table is closed.
 * @return false Table not closed.
 */
const bool observation_table::is_closed() {
  bool is_closed = true;
  for(auto it = lower_table.cbegin(); it != lower_table.cend(); ++it){
    const auto& entry = it->second;
    if(!upper_table_rows.contains(entry)){
      is_closed = false;

      const auto& row = it->first;
      move_to_upper_table(row);
    }
  }

  if(is_closed){
    return true;
  }

  hash_upper_table();
  return false;
}

const vector< pref_suf_t >& observation_table::get_incomplete_rows() const {
  return this->incomplete_rows;
}

void observation_table::mark_row_complete(const pref_suf_t& row) {
  incomplete_rows.erase(row);
}

/**
 * @brief Extends the lower table by adding an extra character to each table. These new rows are by design empty and hence 
 * incomplete_rows is updated.
 * 
 */
void observation_table::extend_lower_table() {
  for(auto it = lower_table.cbegin(); it != lower_table.cend(); ++it){
    auto& row_name = it->first;
    //auto& entry = it->second;

    for(const auto i: alphabet){
      pref_suf_t new_row_name = pref_suf_t(row_name);
      new_row_name.push_back(i);

      incomplete_rows.push_back(pref_suf_t(new_row_name)); // we do a copy to circumvent the destructor
      lower_table[move(new_row_name)] = map<pref_suf_t, knowledge_t>();
    }
  }
}