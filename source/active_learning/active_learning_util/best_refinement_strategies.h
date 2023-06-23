/**
 * @file best_refinement_strategies.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-06-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef BEST_REFINEMENT_STRATEGY_
#define BEST_REFINEMENT_STRATEGY_

#include "refinement.h"
#include "state_merger.h" // TODO: this include is not nice here, only for the node_to_refinement_map_T

namespace best_refinement_strategy {
  refinement* get_refinement(refinement_set* possible_refinements, node_to_refinement_map_T& node_to_refs_map);
};

#endif // BEST_REFINEMENT_