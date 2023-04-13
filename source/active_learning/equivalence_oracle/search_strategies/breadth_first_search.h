/**
 * @file breadth_first_search.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Inefficient.
 * @version 0.1
 * @date 2023-04-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _AL_BREADTH_FIRST_SEARCH_H_
#define _AL_BREADTH_FIRST_SEARCH_H_

#include "search_base.h"

#include <list>
#include <stack>

class bfs_strategy : public search_base {
  private:
    const int BFS_MAX_DEPTH;
    
    int depth;
    std::list<int>::iterator alphabet_it;

    std::stack< list<int> > curr_search;
    std::stack< list<int> > old_search;
  public:
    bfs_strategy(const int max_depth) : search_base(), BFS_MAX_DEPTH(max_depth) {
      depth = 0;
      alphabet_it = nullptr;
    };

    std::optional< const std::vector<int> > next(const std::shared_ptr<sul_base>& sul, const inputdata& id) const override;
};

#endif
