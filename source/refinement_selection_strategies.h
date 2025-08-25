/**
 * @file refinement_selection_strategies.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _REFINEMENT_SELECTION_STRATEGIES_H_
#define _REFINEMENT_SELECTION_STRATEGIES_H_

#include "state_merger.h"
#include "refinement.h"
#include "apta.h"

#include "active_learning/system_under_learning/database_sul.h"

#include <memory>

class selection_strategy_base {
  public: 
    selection_strategy_base() = default;
    virtual refinement* perform(refinement_set* possible_refs, std::shared_ptr<node_to_refinement_map_T> node_to_ref_map) = 0;
};

/**
 * @brief This strategy relies on evidence. If the best possible refinement has more 
 * than one possibility, update all the nodes that it has with the correct statistics. 
 * Then, reevaluate the merges, and if still more than one is possible, we can't do better
 * on this one (statistically speaking) based on the evidence we have. Then perform the 
 * highest scoring one out of the ones we got. 
 * 
 */
class evidence_based_strategy : public selection_strategy_base {
  private:
    std::shared_ptr<database_sul> database_connector;
    state_merger* merger;

  public:
    /**
     * @brief Construct a new evidence based strategy object
     */
    evidence_based_strategy(std::shared_ptr<database_sul>& database_connector, state_merger* merger){
      this->database_connector = database_connector;
      this->merger = merger;
    }

    virtual refinement* perform(refinement_set* possible_refs, std::shared_ptr<node_to_refinement_map_T> node_to_ref_map) override;
};

#endif