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
#include "oracle_base.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"

#include "overlap_fill.h"
#include "overlap_fill_batch_wise.h"
#include "distinguishing_sequence_fill.h"

#include <list>
#include <memory>

#include <future>
#include <mutex>

/**
 * @brief Helper class for us to achieve concurency.
 * 
 * 
 */
class search_instance{
  private:
    // make sure that our underlying structures like the suffix trees do not collide
    std::unique_ptr<ii_base> ii_handler;
    std::unique_ptr<distinguishing_sequences> ds_ptr = std::make_unique<distinguishing_sequences>();
    inline static std::mutex m_mutex;
    const int MIN_BATCH_SIZE = 512;
    const int MAX_LEN = 30;
    
  public:
    search_instance(){
      ii_handler = std::make_unique<distinguishing_sequence_fill>();
    }

    void operator()(std::promise<bool>&& out, std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta, const std::unique_ptr<oracle_base>& oracle, apta_node* red_node, apta_node* blue_node);
};

class paul_algorithm : public algorithm_base {
  protected:
    std::unique_ptr<ii_base> ii_handler;
    refinement* get_best_refinement(std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta);

    /**
     * Relevant for parallelization.
     */
    static bool merge_check(std::unique_ptr<ii_base>& ii_handler, std::unique_ptr<state_merger>& merger, std::unique_ptr<oracle_base>& oracle, std::unique_ptr<apta>& the_apta, apta_node* red_node, apta_node* blue_node);

    std::list<refinement*> retry_merges(std::list<refinement*>& previous_refs, std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta);
    std::list<refinement*> find_hypothesis(std::list<refinement*>& previous_refs, std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta);
    void proc_counterex(inputdata& id, std::unique_ptr<apta>& the_apta, const std::vector<int>& counterex,
                        std::unique_ptr<state_merger>& merger, const refinement_list refs) const;
  public:
    paul_algorithm(std::unique_ptr<oracle_base>&& oracle)
        : algorithm_base(std::move(oracle)){
          ii_handler = std::make_unique<distinguishing_sequence_fill>();
          STORE_ACCESS_STRINGS = true;
        };
    paul_algorithm(std::initializer_list< std::unique_ptr<oracle_base> >&& i_list) : paul_algorithm(std::move(*(i_list.begin()))){
      std::cerr << "This algorithm does not support multiple oracles. Oracle 2 is ignored." << std::endl;
      std::unique_ptr<oracle_base>& ptr = i_list.data()[0];
      paul_algorithm(std::move(ptr));
    }

    void run(inputdata& id) override;
};

#endif
