/**
 * @file probabilistic_lsharp.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The algorithm as described in "PDFA Distillation via String Probability Queries", 
 * Baumgartner and Verwer 2024.
 * @version 0.1
 * @date 2023-10-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _PROBABILISTIC_L_SHARP_H_
#define _PROBABILISTIC_L_SHARP_H_

#include "algorithm_base.h"
#include "lsharp.h"

#include "definitions.h"
#include "base_oracle.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"

#include <list>
#include <memory>
#include <unordered_map>

#if defined(_MSC_VER) 
#  define FLEXFRINGE_ALWAYS_INLINE inline
#else
#  define FLEXFRINGE_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

class probabilistic_lsharp_algorithm : public lsharp_algorithm {
  private:
    bool MAX_DEPTH_REACHED = false;

    FLEXFRINGE_ALWAYS_INLINE void update_tree_dfs(apta* the_apta, const std::vector<int>& alphabet) const;
    FLEXFRINGE_ALWAYS_INLINE void update_access_path(apta_node* n, apta* the_apta,
                                                                  const std::vector<int>& alphabet) const;

    inline void add_statistics(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                               const std::vector<int>& alphabet,
                               std::optional<active_learning_namespace::pref_suf_t> seq_opt) const;

    FLEXFRINGE_ALWAYS_INLINE std::unordered_set<apta_node*>
    extend_fringe_balanced(std::unique_ptr<state_merger>& merger, apta_node* n, std::unique_ptr<apta>& the_apta, inputdata& id,
                  const std::vector<int>& alphabet) const;

    inline void init_final_prob(apta_node* n, apta* the_apta, inputdata& id) const;
    std::list<refinement*> find_complete_base(std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta, inputdata& id,
                                         const std::vector<int>& alphabet);

  protected:
    inline void proc_counterex(inputdata& id, std::unique_ptr<apta>& hypothesis, const std::vector<int>& counterex,
                               std::unique_ptr<state_merger>& merger, const refinement_list refs, const std::vector<int>& alphabet) const;

  public:
    probabilistic_lsharp_algorithm() : lsharp_algorithm() {
      init_standard();      
      STORE_ACCESS_STRINGS = true;
    }

    void run(inputdata& id) override;
};

#endif
