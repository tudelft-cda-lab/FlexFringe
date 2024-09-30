/**
 * @file distinguishing_sequence_fill.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Keeps track of distinguishing sequences, which it then fills up.
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _DISTINGUISHING_SEQUENCES_FILL_H_
#define _DISTINGUISHING_SEQUENCES_FILL_H_

#include "ii_base.h"
#include "distinguishing_sequences.h"

#include <memory>
#include <vector>
#include <list>

class distinguishing_sequence_fill : public ii_base {
  private:
    inline std::vector<int> concat_prefsuf(const std::vector<int>& pref, const std::vector<int>& suff) const;
    inline void add_data_to_tree(std::unique_ptr<apta>& aut, const vector<int>& seq, const string& anwer, const float confidence);

  protected:
    std::unique_ptr<distinguishing_sequences> ds_ptr = std::make_unique<distinguishing_sequences>();

    void add_dist_sequences_to_apta(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right);
    void pre_compute(std::list<int>& suffix, unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right, const int depth);

  public:
    void pre_compute(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right) override;
    void complement_nodes(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right) override;
};

#endif