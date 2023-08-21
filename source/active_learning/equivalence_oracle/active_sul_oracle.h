/**
 * @file active_sul_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This class is for SULs that are queriable. It utilizes a search strategy.
 * @version 0.1
 * @date 2023-04-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _ACTIVE_SUL_ORACLE_H_
#define _ACTIVE_SUL_ORACLE_H_

#include "eq_oracle_base.h"

#include <optional>
#include <utility>

class active_sul_oracle : public eq_oracle_base {
  protected:
    std::shared_ptr<sul_base> sul;
    std::unique_ptr<search_base> search_strategy;

    virtual void reset_sul() override {};

    /**
     * @brief TODO: delete this function from the eq-oracles
     * 
     * @param merger 
     * @param tr 
     * @param id 
     * @return true 
     * @return false 
     */
    [[deprecated]]
    virtual bool apta_accepts_trace(state_merger* merger, const list<int>& tr, inputdata& id) const override {return true;}
  
  public:
    active_sul_oracle(std::shared_ptr<sul_base>& sul) : eq_oracle_base(sul) {
      search_strategy = std::unique_ptr<search_base>(new w_method(35));//std::unique_ptr<search_base>(new bfs_strategy(8)); // number here is maximum length of sequence. Find a better way to set this
      assert(dynamic_cast<input_file_sul*>(sul.get()) == nullptr);
    };

    std::optional< std::pair< std::list<int>, int> > equivalence_query(state_merger* merger, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher);
};

#endif
