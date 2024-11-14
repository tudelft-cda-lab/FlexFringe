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
#include "sul_base.h"

#include "apta.h"
#include "state_merger.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <type_traits> // used in derived classes: check for SUL types

class oracle_base {
  protected:
    std::shared_ptr<sul_base> sul;
    std::unique_ptr<search_base> cex_search_strategy;
    std::unique_ptr<conflict_search_base> conflict_searcher;

    virtual void reset_sul() = 0;

  public:
    oracle_base(std::shared_ptr<sul_base>& sul) : sul(sul){
      cex_search_strategy = cex_search_strategy_factory::create_search_strategy();
    };

    /**
     * @brief Poses the equivalence query. Returns counterexample cex and true answer to cex if no equivalence proven.
     *
     * @param merger The state-merger.
     * @return std::optional< std::pair< std::vector<int>, int> > Counterexample if not equivalent, else nullopt.
     * Counterexample is pair of trace and the answer to the counterexample as returned by the SUL.
     */
    virtual std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger) = 0;

    virtual void initialize(state_merger* merger){
      this->cex_search_strategy->initialize(merger);
    }

    const sul_response ask_sul(const std::vector<int>& query_trace, inputdata& id);
    const sul_response ask_sul(const std::vector<int>&& query_trace, inputdata& id);
    const sul_response ask_sul(const std::vector< std::vector<int> >& query_traces, inputdata& id);
    const sul_response ask_sul(const std::vector< std::vector<int> >&& query_traces, inputdata& id);

    const int ask_membership_query(const active_learning_namespace::pref_suf_t& query, inputdata& id);
    const int ask_membership_query(const active_learning_namespace::pref_suf_t& prefix,
                                   const active_learning_namespace::pref_suf_t& suffix, inputdata& id);
    
    const std::pair<int, float> ask_membership_confidence_query(const active_learning_namespace::pref_suf_t& query, inputdata& id);
    const std::vector< std::pair<int, float> > ask_type_confidence_batch(const std::vector< std::vector<int> >& query_traces, inputdata& id) const;

    
    const std::pair< int, std::vector< std::vector<float> > > get_membership_state_pair(const active_learning_namespace::pref_suf_t& access_seq,
                                                     inputdata& id);
    /* For learning weighted automata or PDFA */
    const double get_string_probability(const active_learning_namespace::pref_suf_t& query, inputdata& id);
    // const float get_symbol_probability(const active_learning_namespace::pref_suf_t& access_seq, const int symbol,
    // inputdata& id);
    const std::vector<float> get_weigth_distribution(const active_learning_namespace::pref_suf_t& access_seq,
                                                     inputdata& id);
    const std::pair< std::vector<float>, std::vector<float> > get_weigth_state_pair(const active_learning_namespace::pref_suf_t& access_seq,
                                                     inputdata& id);


    const int ask_membership_query_maybe(const active_learning_namespace::pref_suf_t& query, inputdata& id);

    const std::unique_ptr<sul_base>& get_sul_ref() const noexcept {
      return sul;
    }
};

#endif
