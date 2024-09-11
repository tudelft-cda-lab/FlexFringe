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

#include <optional>


class overlap_fill : public ii_base {
  private:
    __attribute__((always_inline)) inline void add_data_to_tree(std::unique_ptr<apta>& aut, active_learning_namespace::pref_suf_t& seq, std::unique_ptr<base_teacher>& teacher, apta_node* n, std::optional<int> s_opt=std::nullopt);

  protected:
    const int MAX_DEPTH;

    inline void add_child_node(std::unique_ptr<apta>& aut, apta_node* node, std::unique_ptr<base_teacher>& teacher, const int symbol);

  public:
    overlap_fill(const int MAX_DEPTH=0) : MAX_DEPTH(MAX_DEPTH) {};

    void complement_nodes(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right, const int depth) override;
    void complete_node(apta_node* node, std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher) override;
};

#endif