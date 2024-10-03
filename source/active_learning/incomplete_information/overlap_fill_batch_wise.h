/**
 * @file overlap_fill_batch_wise.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Same as normal overlap fill, but here we collect the sequences first in a batch and then ask the network the whole batch.
 * Harder to follow through in case a more detailed analysis is required, but potentially faster than the normal 
 * implementation.
 * 
 * @version 0.1
 * @date 2024-09-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __OVERLAP_FILL_BATCH_WISE_H__
#define __OVERLAP_FILL_BATCH_WISE_H__

#include "overlap_fill.h"

class overlap_fill_batch_wise : public overlap_fill {
  private:
    const int BATCH_SIZE;
    
    // note: this is the same as in the base class, but to inline we need to copy.
    __attribute__((always_inline)) inline void add_data_to_tree(std::unique_ptr<apta>& aut, const std::vector<int>& seq, const int reverse_type, float confidence, apta_node* node, const int symbol);
    void complement_nodes(std::vector< std::vector<int> >& query_traces, std::vector< std::pair<apta_node*, int> >& query_node_symbol_pairs, std::unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right, const int depth);

  public:
    overlap_fill_batch_wise(const int MAX_DEPTH=0, const int BATCH_SIZE=128) : overlap_fill(MAX_DEPTH), BATCH_SIZE(BATCH_SIZE)  {};

    void complement_nodes(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right) override;
    
    void complete_node(apta_node* node, std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher) override;
};

#endif