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

#include "ii_base.h"
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
class distinguishing_sequence_fill_fast : public ii_base {
  protected:
    const int MIN_BATCH_SIZE = AL_BATCH_SIZE;
    const int MAX_LEN = MAX_AL_SEARCH_DEPTH;

    std::vector< std::vector<int> > m_suffixes;
    std::vector<int> memoized_predictions;
    std::unique_ptr<distinguishing_sequences> ds_ptr = std::make_unique<distinguishing_sequences>();

    void pre_compute(std::list<int>& suffix, std::unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth);
  
    void add_data_to_tree(std::unique_ptr<apta>& aut, const std::vector<int>& seq, const int reverse_type, const float confidence);

  public:
    distinguishing_sequence_fill_fast(const std::shared_ptr<sul_base>& sul) : ii_base(sul){};

    void pre_compute(std::unique_ptr<apta>& aut, apta_node* node) override;
    void pre_compute(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) override;
    //void complement_nodes(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) override;
    bool check_consistency(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) override;

    //const std::vector< std::vector<int> >& get_m_suffixes() noexcept {return m_suffixes;}
    //const std::vector<int>& get_memoized_predictions(){return memoized_predictions;} // TODO: Make this one nicer
    
    const int size() const override {return ds_ptr->size();}
};

#endif
