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

#include <vector>
#include <map>
#include <set>

namespace obs_table_namespace{
  const int EPS = -1; // empty symbol special character. flexfringe does not map to -1 by design.

  typedef std::vector<int> pref_suf;

  enum class upper_lower_t{
    upper,
    lower
  };

  enum class knowledge_t{
    accepting,
    rejecting,
    unknown
  };
}
 

class observation_table{
  protected:
    const std::vector<int> alphabet;

    std::set<obs_table_namespace::pref_suf> all_colums;

    std::map< obs_table_namespace::pref_suf, obs_table_namespace::upper_lower_t> table_mapper; // prefix in upper table or lower table?

    // two dimensional maps
    std::map< obs_table_namespace::pref_suf, std::map<obs_table_namespace::pref_suf,obs_table_namespace::knowledge_t> > upper_table; 
    std::map< obs_table_namespace::pref_suf, std::map<obs_table_namespace::pref_suf,obs_table_namespace::knowledge_t> > lower_table; 

    void extend_lower_table();
    
    obs_table_namespace::pref_suf get_null_vector() const noexcept {
      return obs_table_namespace::pref_suf{obs_table_namespace::EPS};
    }

  public:
    observation_table(std::vector<int> alphabet);

    const obs_table_namespace::knowledge_t get_answer(const pref_suf test_string) const;
};

#endif
