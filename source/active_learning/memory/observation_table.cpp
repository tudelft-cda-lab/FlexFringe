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

#include <utility>
#include <stdexcept>

using namespace std;
using namespace obs_table_namespace;

void observation_table::extend_lower_table() {
  for(auto it = lower_table.cbegin(); it != lower_table.cend(); ++it){
    auto& row_name = it->first;
    //auto& entry = it->second;

    for(const auto i: alphabet){
      pref_suf new_row_name = pref_suf(row_name);
      new_row_name.push_back(i);
      lower_table[move(new_row_name)] = map<pref_suf, knowledge_t>();
    }

    //for(const auto& col_name: all_colums){
    //  auto new_col_name = pref_suf(col_name);
    //  
    //}
  }
}

const knowledge_t observation_table::get_answer(const pref_suf test_string) const {
  if(table_mapper.count(test_string) == 0){
    throw logic_error("This case is not yet implemented. What shall we do?");
  }

  auto table_select = table_mapper[test_string];
  switch(table_select){
    case upper_lower_t::upper:
      //
    case upper_lower_t::lower:
      //
    default:
      throw runtime_error("Unknown table_select variable occured in observation_table::get_answer.");
  }

  // TODO: finish implementation of get_answer
}

observation_table::observation_table(vector<int> alphabet) : alphabet(alphabet.begin(), alphabet.end()) {
  //alphabet.push_back(EPS);

  const auto nullvector = get_null_vector();
  upper_table[nullvector][nullvector] = knowledge_t::accepting;

  table_mapper[nullvector] = upper_lower_t::upper;

  extend_lower_table();
};