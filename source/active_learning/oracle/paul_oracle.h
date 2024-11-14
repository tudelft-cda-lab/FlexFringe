/**
 * @file paul_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _PAUL_ORACLE_H_
#define _PAUL_ORACLE_H_

#include "active_sul_oracle.h"
#include "parameters.h"

#include "linear_conflict_search.h"

#include <optional>
#include <utility>

class paul_oracle : public active_sul_oracle {
  protected:
    int get_sul_response(const std::vector< std::vector<int> >& query_string, inputdata& id) const;
    void reset_sul() override {};

  public:
    paul_oracle(std::shared_ptr<sul_base>& sul) : active_sul_oracle(sul) {
        cex_search_strategy = std::unique_ptr<search_base>(
            //new random_w_method(MAX_AL_SEARCH_DEPTH)); // std::unique_ptr<search_base>(new bfs_strategy(8)); // number here is
            new random_w_method(MAX_AL_SEARCH_DEPTH)); // std::unique_ptr<search_base>(new bfs_strategy(8)); // number here is
                                           // maximum length of sequence. Find a better way to set this

        conflict_searcher = std::unique_ptr<conflict_search_base>(new dfa_conflict_search_namespace::linear_conflict_search());
        assert(dynamic_cast<input_file_sul*>(sul.get()) == nullptr);
    };

    std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger);
};

#endif
