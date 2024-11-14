/**
 * @file string_probability_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This class is for SULs that are queriable. It utilizes a search strategy.
 * @version 0.1
 * @date 2023-04-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _STRING_PROBABILITY_ORACLE_H_
#define _STRING_PROBABILITY_ORACLE_H_

#include "oracle_base.h"
#include "state_merger.h"
#include "parameters.h"

#include <optional>
#include <utility>

class string_probability_oracle : public oracle_base {
  protected:
    state_merger* merger;

    virtual void reset_sul() override{};

  public:
    string_probability_oracle(std::unique_ptr<sul_base>& sul) : oracle_base(sul) {
        cex_search_strategy = std::unique_ptr<search_base>(new random_string_search(MAX_AL_SEARCH_DEPTH));
        // cex_search_strategy = std::unique_ptr<search_base>(new bfs_strategy(8)); // number here is maximum length of
        // sequence. Find a better way to set this
        assert(dynamic_cast<input_file_sul*>(sul.get()) == nullptr);

        merger = nullptr;
    };

    std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger);
};

#endif
