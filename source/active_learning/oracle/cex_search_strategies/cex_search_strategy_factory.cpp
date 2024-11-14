/**
 * @file search_strategy_factory.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "cex_search_strategy_factory.h"

#include "random_w_method.h"
#include "fringe_walk.h"
#include "breadth_first_search.h"
#include "targeted_bfs_walk.h"
#include "random_string_search.h"

using namespace std;

/**
 * @brief Does what you think it does.
 */
unique_ptr<search_base> cex_search_strategy_factory::create_search_strategy() {
  if(CEX_SEARCH_STRATEGY == "random_w_method")
    return make_unique<random_w_method>();
  else if(CEX_SEARCH_STRATEGY == "fringe_walk")
    return make_unique<fringe_walk>();
  else if(CEX_SEARCH_STRATEGY == "breadth_first_search")
    return make_unique<breadth_first_search>();
  else if(CEX_SEARCH_STRATEGY == "targeted_bfs_walk")
    return make_unique<targeted_bfs_walk>();
  else if(CEX_SEARCH_STRATEGY == "random_string_search")
    return make_unique<random_string_search>();
  else
    throw invalid_argument("cex_search_strategy input argument not recognized.");
}