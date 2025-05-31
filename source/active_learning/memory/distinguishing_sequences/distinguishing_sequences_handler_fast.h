/**
 * @file distinguishing_sequences_handler_fast.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Keeps track of distinguishing sequences by storing them in raw format. 
 * As the name implies, this version is fast, but consumes more memory at runtime. Only use 
 * when you don't expect the size of the distinguishing sequences to fit into working memory.
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _DISTINGUISHING_SEQUENCES_HANDLER_FAST_H_
#define _DISTINGUISHING_SEQUENCES_HANDLER_FAST_H_

#include "distinguishing_sequences_handler.h"
#include "data_structures/distinguishing_sequences.h"
#include "parameters.h"

#include <memory>
#include <vector>
#include <list>
#include <unordered_set>

/**
 * @brief Works the same as the distinguishing_sequences_handler_fast, but it memoizes the sequences in a list as well. 
 * This way the sequences do not have to be reconstructed in every turn. The trade-off of this is obviously 
 * it requires more working memory.
 */
class distinguishing_sequences_handler_fast final : public distinguishing_sequences_handler {
  private:
    layerwise_suffixes_t m_suffixes;

  protected:
    inline void pre_compute(std::list<int>& suffix, std::unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth) override;

    // we may kick out those two methods, as the layer-wise equivalents below serve our purpose better
    [[maybe_unused]] std::vector<int> predict_node_with_automaton(apta& aut, apta_node* node) override;
    [[maybe_unused]] std::vector<int> predict_node_with_sul(apta& aut, apta_node* node) override;

    layer_predictions_map predict_node_with_automaton_layer_wise(apta& aut, apta_node* node) override; 
    layer_predictions_map predict_node_with_sul_layer_wise(apta& aut, apta_node* node) override;

    const int size() const override {return m_suffixes.size();}

    bool distributions_consistent_layer_wise(const layer_predictions_map& v1, 
                                             const layer_predictions_map& v2,
                                             const std::optional<int> depth1_opt = std::nullopt,
                                             const std::optional<int> depth2_opt = std::nullopt) 
                                             override;    

    float compute_threshold(const std::optional<int>& d1, const std::optional<int>& d2); 

  public:
    distinguishing_sequences_handler_fast(const std::shared_ptr<sul_base>& sul) : distinguishing_sequences_handler(sul){};
    void add_suffix(const std::vector<int>& seq) override {m_suffixes.add_suffix(seq);} 
};

#endif
