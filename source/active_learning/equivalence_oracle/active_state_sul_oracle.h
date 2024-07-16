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

#include "eq_oracle_base.h"
#include "parameters.h"
#include "linear_conflict_search.h"

#include <optional>
#include <utility>

class active_state_sul_oracle : public eq_oracle_base {
  protected:
    std::shared_ptr<sul_base> sul;
    std::unique_ptr<search_base> search_strategy;

    virtual void reset_sul() override{};

  public:
    active_state_sul_oracle(std::shared_ptr<sul_base>& sul) : eq_oracle_base(sul) {
        search_strategy = std::unique_ptr<search_base>(
            new random_string_search(MAX_CEX_LENGTH)); // std::unique_ptr<search_base>(new bfs_strategy(8)); // number here is
                                           // maximum length of sequence. Find a better way to set this
        assert(dynamic_cast<input_file_sul*>(sul.get()) == nullptr);
        conflict_searcher = std::unique_ptr<conflict_search_base>(new linear_conflict_search());
    };

    std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher);
};

#endif
