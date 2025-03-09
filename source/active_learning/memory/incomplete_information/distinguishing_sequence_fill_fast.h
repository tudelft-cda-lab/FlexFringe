/**
 * @file distinguishing_sequence_fill_fast.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Keeps track of distinguishing sequences, which it then fills up.
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _DISTINGUISHING_SEQUENCES_FILL_FAST_H_
#define _DISTINGUISHING_SEQUENCES_FILL_FAST_H_

#include "distinguishing_sequence_fill.h"
#include "distinguishing_sequences.h"
#include "parameters.h"

#include <memory>
#include <vector>
#include <list>
#include <unordered_set>

/**
 * @brief Works the same as the distinguishing_sequence_fill_fast, but it memoizes the sequences in a list as well. 
 * This way the sequences do not have to be reconstructed in every turn. The trade-off of this is obviously 
 * it requires more working memory.
 */
class distinguishing_sequence_fill_fast : public distinguishing_sequence_fill {
  protected:

    std::vector< std::vector<int> > m_suffixes;

    void pre_compute(std::list<int>& suffix, std::unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth) override;

    std::vector<int> predict_node_with_automaton(apta& aut, apta_node* node) override;
    std::vector<int> predict_node_with_sul(apta& aut, apta_node* node) override;

  public:
    distinguishing_sequence_fill_fast(const std::shared_ptr<sul_base>& sul) : distinguishing_sequence_fill(sul){};
};

#endif
