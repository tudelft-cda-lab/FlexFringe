/**
 * @file observation_table.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _OBS_TABLE_H_
#define _OBS_TABLE_H_

#include "definitions.h"

#include <list>
#include <map>
#include <set>
#include <list>

namespace obs_table_namespace{
  enum class upper_lower_t{
    upper,
    lower
  };

  typedef std::map< active_learning_namespace::pref_suf_t, int > row_type; // reference is all_columns
  typedef std::map< active_learning_namespace::pref_suf_t, row_type > table_type;
}
 

class observation_table{
  protected:
    bool checked_for_closedness;
    const std::vector<int> alphabet;
    std::set<active_learning_namespace::pref_suf_t> all_columns;
    std::map< active_learning_namespace::pref_suf_t, obs_table_namespace::upper_lower_t> table_mapper; // decides if prefix in upper table or lower table
    std::list< active_learning_namespace::pref_suf_t > incomplete_rows;

    // the actual table
    obs_table_namespace::table_type upper_table; 
    obs_table_namespace::table_type lower_table; 

    // keeping easier track of all the rows to check for closedness
    std::set<obs_table_namespace::row_type> upper_table_rows;
    void hash_upper_table();
    void move_to_upper_table(const active_learning_namespace::pref_suf_t& row);

    active_learning_namespace::pref_suf_t get_null_vector() const noexcept {
      return active_learning_namespace::pref_suf_t();
    }

    const active_learning_namespace::pref_suf_t map_prefix(const active_learning_namespace::pref_suf_t& column) const;

    const bool record_is_in_selected_table(const obs_table_namespace::table_type& selected_table, const active_learning_namespace::pref_suf_t& row, const active_learning_namespace::pref_suf_t& col) const;
    void insert_record_in_selected_table(obs_table_namespace::table_type& selected_table, const active_learning_namespace::pref_suf_t& row, const active_learning_namespace::pref_suf_t& col, const int answer);
    int get_answer_from_selected_table(const obs_table_namespace::table_type& selected_table, const active_learning_namespace::pref_suf_t& row, const active_learning_namespace::pref_suf_t& col) const;
    
  public:
    observation_table() = delete;
    observation_table(const std::vector<int>& alphabet);

    const bool has_record(const active_learning_namespace::pref_suf_t& row, const active_learning_namespace::pref_suf_t& col) const;
    void insert_record(const active_learning_namespace::pref_suf_t& row, const active_learning_namespace::pref_suf_t& col, const int answer);
    int get_answer(const active_learning_namespace::pref_suf_t& row, const active_learning_namespace::pref_suf_t& col) const;

    const bool is_closed();

    const obs_table_namespace::table_type& get_upper_table() const noexcept {
      return upper_table;
    }

    const obs_table_namespace::table_type& get_lower_table() const noexcept {
      return lower_table;
    }
    
    void extend_lower_table();
    const std::list< active_learning_namespace::pref_suf_t >& get_incomplete_rows() const;
    void mark_row_complete(const active_learning_namespace::pref_suf_t& row);
    void extent_columns(const active_learning_namespace::pref_suf_t& suffix);

    const auto& get_column_names() const noexcept {
      return all_columns;
    }

    [[maybe_unused]]
    void print() const;
};

#endif