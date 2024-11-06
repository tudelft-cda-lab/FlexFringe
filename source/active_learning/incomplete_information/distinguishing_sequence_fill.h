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
#include <unordered_set>

class distinguishing_sequence_fill : public ii_base {
  private:
    inline std::vector<int> concat_prefsuf(const std::vector<int>& pref, const std::vector<int>& suff) const;
    inline void add_data_to_tree(std::unique_ptr<apta>& aut, const std::vector<int>& seq, const int reverse_type, const float confidence);

  protected:
    const int MIN_BATCH_SIZE = 256;
    const int MAX_LEN = 30;

    inline static std::vector< std::vector<int> > m_suffixes;
    inline static std::vector<int> memoized_predictions; // static for multithreading
    std::unique_ptr<distinguishing_sequences> ds_ptr = std::make_unique<distinguishing_sequences>();

    void add_dist_sequences_to_apta(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right);
    void pre_compute(std::list<int>& suffix, std::unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right, const int depth);
    void pre_compute(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* node) override;
  
  public:
    void pre_compute(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right) override;
    void complement_nodes(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right) override;
    bool check_consistency(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right) override;

    void memoize() noexcept override;
    const std::vector< std::vector<int> >& get_m_suffixes() noexcept {return m_suffixes;}

    const std::vector<int>& get_memoized_predictions(){return memoized_predictions;} // TODO: Make this one nicer
    const int size() const override {return ds_ptr->size();}
};

#endif
