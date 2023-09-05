/**
 * @file probabilistic_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This class is for SULs that are queriable. It utilizes a search strategy.
 * @version 0.1
 * @date 2023-04-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _PROBABILISTIC_ORACLE_H_
#define _PROBABILISTIC_ORACLE_H_

#include "eq_oracle_base.h"

#include <optional>
#include <utility>

class probabilistic_oracle : public eq_oracle_base {
  protected:
    const double max_distance = 0.05; // we assume the networks are good at their predictions

    std::shared_ptr<sul_base> sul;
    std::unique_ptr<search_base> search_strategy;
    std::shared_ptr< unordered_map<apta_node*, unordered_map<int, double> > >  node_response_map;

    virtual void reset_sul() override {};
  
  public:
    probabilistic_oracle(std::shared_ptr<sul_base>& sul) : eq_oracle_base(sul) {
      search_strategy = std::unique_ptr<search_base>(new random_string_search(30));//std::unique_ptr<search_base>(new bfs_strategy(8)); // number here is maximum length of sequence. Find a better way to set this
      assert(dynamic_cast<input_file_sul*>(sul.get()) == nullptr);
    };

    std::optional< std::pair< std::vector<int>, int> > equivalence_query(state_merger* merger, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher);
};

#endif
