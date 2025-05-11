/**
 * @file paul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-08-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _PAUL_H_
#define _PAUL_H_

#include "algorithm_base.h"

#include "definitions.h"
#include "base_oracle.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"

#include "paul_heuristic.h"
#include "ds_handler_factory.h"

#include <list>
#include <unordered_set>
#include <memory>

#include <future>
#include <mutex>

class paul_algorithm final : public algorithm_base {    
  private:
    std::shared_ptr<distinguishing_sequences_handler_base> ds_handler;
    const bool MERGE_WITH_LARGEST = true;

    /* inline */ void update_node_data(apta_node* n, std::unique_ptr<apta>& aut) const;
    /* inline */ paul_data* get_node_data(apta_node* n) const;

    void complete_node(apta_node* n, std::unique_ptr<state_merger>& merger) const;
    void create_child_node(apta_node* parent_node, std::unique_ptr<state_merger>& merger, const std::vector<int>& seq, inputdata& id) const;

    refinement* get_best_refinement(std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta);
    refinement* check_blue_node_for_merge_partner(apta_node* const blue_node, std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta,
                                                  const state_set& red_its);

    void load_inputdata();

    std::list<refinement*> retry_merges(std::list<refinement*>& previous_refs, std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta);
    std::list<refinement*> find_hypothesis(std::list<refinement*>& previous_refs, std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta);
    void proc_counterex(inputdata& id, std::unique_ptr<apta>& the_apta, const std::vector<int>& counterex,
                        std::unique_ptr<state_merger>& merger, const refinement_list refs) const;
  public:
    paul_algorithm(){
      auto sul = sul_factory::create_sul(AL_SYSTEM_UNDER_LEARNING);
      this->ds_handler = ds_handler_factory::create_ds_handler(sul, AL_II_NAME);
      this->oracle = oracle_factory::create_oracle(sul, AL_ORACLE, this->ds_handler);

      STORE_ACCESS_STRINGS = true;
      load_inputdata();
    }

    void run(inputdata& id) override;
};

#endif
