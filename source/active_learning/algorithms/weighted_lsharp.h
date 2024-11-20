/**
 * @file weighted_lsharp.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-11-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _WEIGHTED_L_SHARP_H_
#define _WEIGHTED_L_SHARP_H_

#include "algorithm_base.h"
#include "lsharp.h"

#include "definitions.h"
#include "oracle_base.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"
#include "weight_comparing_oracle.h"

#include <list>
#include <memory>
#include <unordered_map>

class weighted_lsharp_algorithm : public lsharp_algorithm {
  private:
    const bool use_sinks;
  protected:
    void proc_counterex(inputdata& id, std::unique_ptr<apta>& hypothesis, const std::vector<int>& counterex, 
                        std::unique_ptr<state_merger>& merger, const refinement_list refs, const std::vector<int>& alphabet) const;

    void extend_fringe(std::unique_ptr<state_merger>& merger, apta_node* n,
                                                 std::unique_ptr<apta>& the_apta, inputdata& id,
                                                 const std::vector<int>& alphabet) const;
    void query_weights(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                               const std::vector<int>& alphabet,
                               std::optional<active_learning_namespace::pref_suf_t> seq_opt) const;

    std::list<refinement*> find_complete_base(std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta, inputdata& id,
                                         const std::vector<int>& alphabet) override;

  public:
    weighted_lsharp_algorithm(std::unique_ptr<oracle_base>&& oracle) : lsharp_algorithm(std::move(oracle)), use_sinks(USE_SINKS) {
        if(dynamic_cast<weight_comparing_oracle*>(oracle.get()) == nullptr)
          throw std::logic_error("Weighted L# only works with weight_comparing_oracle");
    };

    void run(inputdata& id) override;
};

#endif
