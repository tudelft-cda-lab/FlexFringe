/**
 * @file breadth_first_search.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief BFS search. Not practical in real scenarios, but useful for a deterministic search to look for specific issues in algorithms.
 * @version 0.1
 * @date 2023-04-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _AL_BREADTH_FIRST_SEARCH_H_
#define _AL_BREADTH_FIRST_SEARCH_H_

#include "search_base.h"

#include <stack>

class bfs_strategy : public search_base {
  private:

    std::vector<int> get_vector_from_indices(const std::vector<int>& alphabet, const bool redo) const;
    std::vector<int> get_next_vector(const std::vector<int>& alphabet);

    std::vector<int> vector_idxs; // stores the indices of the respective characters of the alphabet. Content goes from [0, alphabet.size()-1]

  public:
    bfs_strategy(const int max_depth) : search_base(max_depth) { };

    std::optional<std::vector<int>> next(const inputdata& id) override;
};

#endif
