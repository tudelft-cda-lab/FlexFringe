/**
 * @file observation_table_imat.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _OBS_TABLE_IMAT_H_
#define _OBS_TABLE_IMAT_H_

#include "base_teacher.h"
#include "definitions.h"
#include "inputdata.h"

#include <list>
#include <map>
#include <set>
#include <unordered_map>

namespace obs_table_imat_namespace {
enum class upper_lower_t { upper, lower };

typedef std::vector<int> row_type; // reference is all_columns
typedef std::map<active_learning_namespace::pref_suf_t, row_type> table_type;
} // namespace obs_table_imat_namespace

class observation_table_imat {
  protected:
    bool checked_for_closedness;
    const std::vector<int> alphabet;
    std::set<active_learning_namespace::pref_suf_t> all_columns;
    std::map<active_learning_namespace::pref_suf_t, obs_table_imat_namespace::upper_lower_t>
        table_mapper; // decides if prefix in upper table or lower table
    std::list<active_learning_namespace::pref_suf_t> incomplete_rows;

    // the actual table
    obs_table_imat_namespace::table_type upper_table;
    obs_table_imat_namespace::table_type lower_table;

    // keeping easier track of all the rows to check for closedness
    void move_to_upper_table(const active_learning_namespace::pref_suf_t& row);

    active_learning_namespace::pref_suf_t get_null_vector() const noexcept {
        return active_learning_namespace::pref_suf_t();
    }

    const active_learning_namespace::pref_suf_t map_prefix(const active_learning_namespace::pref_suf_t& column) const;

    void insert_record_in_selected_table(obs_table_imat_namespace::table_type& selected_table,
                                         const active_learning_namespace::pref_suf_t& row,
                                         const active_learning_namespace::pref_suf_t& col, const int answer);
    int get_answer_from_selected_table(const obs_table_imat_namespace::table_type& selected_table,
                                       const active_learning_namespace::pref_suf_t& row,
                                       const active_learning_namespace::pref_suf_t& col) const;

  public:
    observation_table_imat() = delete;
    observation_table_imat(const std::vector<int>& alphabet);

    // Maintain the indexes of the columns in the row_type.
    int new_ind = 0;
    std::map<active_learning_namespace::pref_suf_t, size_t> exp2ind;
    void complete_rows(base_teacher* teacher, inputdata& id);
    const bool has_record(const active_learning_namespace::pref_suf_t& row,
                          const active_learning_namespace::pref_suf_t& col);
    void insert_record(const active_learning_namespace::pref_suf_t& row,
                       const active_learning_namespace::pref_suf_t& col, const int answer);
    int get_answer(const active_learning_namespace::pref_suf_t& row,
                   const active_learning_namespace::pref_suf_t& col) const;

    const bool is_closed();

    const obs_table_imat_namespace::table_type& get_upper_table() const noexcept { return upper_table; }

    const obs_table_imat_namespace::table_type& get_lower_table() const noexcept { return lower_table; }

    void extend_lower_table();
    const std::list<active_learning_namespace::pref_suf_t>& get_incomplete_rows() const;
    void mark_row_complete(const active_learning_namespace::pref_suf_t& row);
    void extent_columns(const active_learning_namespace::pref_suf_t& suffix);

    const auto& get_column_names() const noexcept { return all_columns; }

    [[maybe_unused]] void print() const;
};

#endif
