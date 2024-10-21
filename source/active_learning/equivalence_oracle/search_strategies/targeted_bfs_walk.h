/**
 * @file targeted_bfs_walk.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-06-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _TARGETED_BFS_SEARCH_H_
#define _TARGETED_BFS_SEARCH_H_

#include "random_int_generator.h"
#include "search_base.h"

#include <random>
#include <vector>
#include <unordered_set>
#include <queue>

class targeted_bfs_walk : public search_base {
  private:
    apta* hypothesis; // must point to the same node that the oracle also points to
    std::vector<int> alphabet;

    apta_node* current_node;
    std::vector<int> current_access_sequence;
    std::queue<apta_node*> node_queue;

    std::unordered_set<apta_node*> tested_nodes; // nodes that have been extensively tested
    std::unordered_set<apta_node*> queued_nodes; // nodes that have been visited since last call of initialize()/reset()

    const int SAMPLES_PER_NODE = NUM_CEX_PARAM;
    int samples_for_current_node;
    int n_tested_nodes;
    int current_h_size;

    random_int_generator length_generator;
    random_int_generator alphabet_sampler;

    int count_nodes();
    void add_nodes_to_queue(apta_node* node);

  public:
    targeted_bfs_walk(const int max_depth) : search_base(max_depth), current_node(nullptr) {
        length_generator.set_limits(1, max_depth);
        samples_for_current_node = 0;
    };

    std::optional<std::vector<int>> next(const inputdata& id) override;

    void reset() override;

    void initialize(state_merger* merger) override;
};

#endif
