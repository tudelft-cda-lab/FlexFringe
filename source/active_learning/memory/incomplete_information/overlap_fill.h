/**
 * @file overlap_fill.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This one looks for the overlap and completes information on one or the other node if necessary.
 * For example, take the EDSM algorithm. It works by the overlap that actually exists. What if we could ask those missing pieces?
 * 
 * Used in PAUL.
 * 
 * @version 0.1
 * @date 2024-08-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __OVERLAP_FILL_H__
#define __OVERLAP_FILL_H__

#include "ii_base.h"
#include "parameters.h"

#include <optional>
#include <unordered_set>


class overlap_fill : public ii_base {
  private:
    __attribute__((always_inline)) inline void add_data_to_tree(std::unique_ptr<apta>& aut, active_learning_namespace::pref_suf_t& seq, apta_node* n, std::optional<int> s_opt=std::nullopt);

  protected:
    const int MAX_DEPTH;

    inline void add_child_node(std::unique_ptr<apta>& aut, apta_node* node, const int symbol);
    void complement_nodes(std::unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth);
    void complete_node(apta_node* node, std::unique_ptr<apta>& aut);

  public:
    overlap_fill(const std::shared_ptr<sul_base>& sul) : ii_base(sul), MAX_DEPTH(AL_MAX_SEARCH_DEPTH) {};

    void pre_compute(std::unique_ptr<apta>& aut, apta_node* left) override { /*Nothing to do here*/ };
    void pre_compute(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) override;
    bool check_consistency(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) override { return true; }
};

#endif