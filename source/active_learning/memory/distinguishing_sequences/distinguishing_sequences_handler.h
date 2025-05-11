/**
 * @file distinguishing_sequences_handler.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Keeps track of distinguishing sequences, which it then fills up.
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _DISTINGUISHING_SEQUENCES_HANDLER_H_
#define _DISTINGUISHING_SEQUENCES_HANDLER_H_

#include "distinguishing_sequences_handler_base.h"
#include "data_structures/distinguishing_sequences.h"
#include "parameters.h"

#include <memory>
#include <vector>
#include <list>
#include <unordered_set>

class distinguishing_sequences_handler : public distinguishing_sequences_handler_base, 
                                     public std::enable_shared_from_this<distinguishing_sequences_handler>{
  
  friend class paul_algorithm;

  private:
    std::unique_ptr<distinguishing_sequences> ds_ptr = std::make_unique<distinguishing_sequences>();
  
  protected:
    const int MIN_BATCH_SIZE = AL_BATCH_SIZE;
    const int MAX_LEN = AL_MAX_SEARCH_DEPTH;
    const int LTD_LENGTH = -1; // window size of long term dependency

    double last_overlap = 0;

    std::vector<int> memoized_predictions;

    virtual void pre_compute(std::list<int>& suffix, std::unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth);
  
    void add_data_to_tree(std::unique_ptr<apta>& aut, const std::vector<int>& seq, const int reverse_type, const float confidence);
    
    std::vector<int> predict_node_with_automaton(apta& aut, apta_node* node) override;
    std::vector<int> predict_node_with_sul(apta& aut, apta_node* node) override;
    bool distributions_consistent(const std::vector<int>& v1, const std::vector<int>& v2) override;

  public:
    distinguishing_sequences_handler(const std::shared_ptr<sul_base>& sul) 
    : distinguishing_sequences_handler_base(sul), MIN_BATCH_SIZE(AL_BATCH_SIZE), MAX_LEN(AL_MAX_SEARCH_DEPTH) {};

    [[nodiscard]] const bool collect_suffixes() const;

    void initialize(std::unique_ptr<apta>& aut) override;
    void pre_compute(std::unique_ptr<apta>& aut, apta_node* node) override;
    void pre_compute(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) override;
    bool check_consistency(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) override;

    double get_score() override;

    const int size() const override {return ds_ptr->size();}
    
    virtual void add_suffix(const std::vector<int>& seq);
};

#endif
