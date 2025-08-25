/**
 * @file random_w_method.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-05-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_W_METHOD_SEARCH_H_
#define _AL_W_METHOD_SEARCH_H_

#include "random_int_generator.h"
#include "search_base.h"

#include <random>
#include <vector>
#include <unordered_set>

class random_w_method : public search_base {
  private:
    apta* hypothesis; // must point to the same node that the oracle also points to
    red_state_iterator r_it;
    std::vector<int> current_access_sequence;

    std::unordered_set<apta_node*> tested_nodes;
  
    const int SAMPLES_PER_NODE = AL_NUM_CEX_PARAM;
    int samples_for_current_node;
    int n_tested_nodes;
    int current_h_size;

    random_int_generator length_generator;
    random_int_generator alphabet_sampler;

    std::vector<int> alphabet_vec;

    int count_nodes();

  public:
    random_w_method() : r_it(nullptr){
      length_generator.set_limits(1, MAX_SEARCH_DEPTH);
      samples_for_current_node = 0;
    };

    std::optional<std::vector<int>> next(const inputdata& id) override;

    void reset() override;

    void initialize(state_merger* merger) override;
};

#endif
