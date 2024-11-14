/**
 * @file fringe_walk.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-05-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _FRINGE_WALK_SEARCH_H_
#define _FRINGE_WALK_SEARCH_H_

#include "random_int_generator.h"
#include "search_base.h"

#include <random>
#include <vector>
#include <unordered_set>

class fringe_walk : public search_base {
  private:
    apta* hypothesis; // must point to the same node that the oracle also points to
    apta_node* current_node;
    
    std::vector<int> current_access_sequence;
    std::unordered_set<apta_node*> collect_fringe_nodes();
    apta_node* get_next_valid_node(const std::unordered_set<apta_node*>& fringe_nodes, const std::unordered_set<apta_node*>& tested_nodes) const;

    std::unordered_set<apta_node*> tested_nodes;
    std::unordered_set<apta_node*> fringe_nodes;

    const int SAMPLES_PER_NODE = NUM_CEX_PARAM;
    int samples_for_current_node;
    int n_tested_nodes;
    int current_h_size;

    random_int_generator length_generator;
    random_int_generator alphabet_sampler;

    std::vector<int> alphabet_vec;

  public:
    fringe_walk(const int max_depth) : search_base(max_depth){
        length_generator.set_limits(1, max_depth);
        samples_for_current_node = 0;
    };

    std::optional<std::vector<int>> next(const inputdata& id) override;

    void reset() override;

    void initialize(state_merger* merger) override;
};

#endif
