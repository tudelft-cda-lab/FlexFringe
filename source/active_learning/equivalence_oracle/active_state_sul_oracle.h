/**
 * @file active_state_sul_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-06-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _ACTIVE_STATE_SUL_ORACLE_H_
#define _ACTIVE_STATE_SUL_ORACLE_H_

#include "active_sul_oracle.h"
#include "parameters.h"
#include "linear_state_query_conflict_search.h"

#include <optional>
#include <utility>

class active_state_sul_oracle : public active_sul_oracle {
  protected:
    int get_teacher_response(const std::vector<int>& query_string, const std::unique_ptr<base_teacher>& teacher, inputdata& id) const override;

  public:
    active_state_sul_oracle(std::shared_ptr<sul_base>& sul) : active_sul_oracle(sul) {
        //search_strategy = std::unique_ptr<search_base>(
            //new random_w_method(MAX_AL_SEARCH_DEPTH)); // std::unique_ptr<search_base>(new bfs_strategy(8)); // number here is
                                           // maximum length of sequence. Find a better way to set this
        search_strategy = std::unique_ptr<search_base>(new bfs_strategy(MAX_AL_SEARCH_DEPTH));
        assert(dynamic_cast<input_file_sul*>(sul.get()) == nullptr);
        conflict_searcher = std::unique_ptr<conflict_search_base>(new dfa_conflict_search_namespace::linear_state_query_conflict_search());
    };
};

#endif
