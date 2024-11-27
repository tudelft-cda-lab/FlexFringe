/**
 * @file oracle_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _ORACLE_BASE_H_
#define _ORACLE_BASE_H_

#include "parameters.h"
#include "cex_search_strategies/cex_search_strategy_factory.h" // TODO: won't eat those two without the path. Why?
#include "cex_conflict_search/conflict_search_base.h" // TODO: won't eat those two without the path. Why?
#include "conflict_detectors/conflict_detector_base.h" // TODO: won't eat those two without the path. Why?
#include "sul_base.h"

#include "apta.h"
#include "state_merger.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

class oracle_base {
  protected:
    std::shared_ptr<sul_base> sul;

    std::unique_ptr<search_base> cex_search_strategy;
    std::unique_ptr<conflict_search_base> conflict_searcher; // these two are to be determined by derived classes
    std::shared_ptr<conflict_detector_base> conflict_detector; // these two are to be determined by derived classes

    virtual void reset_sul() = 0;

  public:
    oracle_base(const std::shared_ptr<sul_base>& sul) : sul(sul) {
      cex_search_strategy = cex_search_strategy_factory::create_search_strategy();
    }

    // For logic behind this see e.g. https://stackoverflow.com/a/10001573/11956515
    oracle_base(){
      throw std::logic_error("oracle_base call to overloaded constructor providing the sul has to be called!");
    }

    virtual void initialize(state_merger* merger);

    const sul_response ask_sul(const std::vector<int>& query_trace, inputdata& id) const;
    const sul_response ask_sul(const std::vector<int>& prefix, const std::vector<int>& suffix, inputdata& id) const;
    const sul_response ask_sul(const std::vector<int>&& query_trace, inputdata& id) const;
    const sul_response ask_sul(const std::vector<int>&& prefix, const std::vector<int>&& suffix, inputdata& id) const;
    const sul_response ask_sul(const std::vector< std::vector<int> >& query_traces, inputdata& id) const;
    const sul_response ask_sul(const std::vector< std::vector<int> >&& query_traces, inputdata& id) const;

    virtual std::optional<std::pair<std::vector<int>, sul_response>>
    equivalence_query(state_merger* merger);
};

#endif
