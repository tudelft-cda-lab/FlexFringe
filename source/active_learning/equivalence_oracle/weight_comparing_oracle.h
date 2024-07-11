/**
 * @file weight_comparing_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This class is for SULs that are queriable. It utilizes a search strategy.
 * @version 0.1
 * @date 2023-04-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _WEIGHT_COMPARING_ORACLE_H_
#define _WEIGHT_COMPARING_ORACLE_H_

#include "eq_oracle_base.h"
#include "state_merger.h"
#include "parameters.h"

#include <optional>
#include <utility>

class weight_comparing_oracle : public eq_oracle_base {
  private:
    const bool use_sinks; 

  protected:
    std::shared_ptr<sul_base> sul;
    std::unique_ptr<search_base> search_strategy;
    state_merger* merger;

    virtual void reset_sul() override{};

    bool
    test_trace_accepted(apta& hypothesis, trace* const tr, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher, inputdata& id);

  public:
    weight_comparing_oracle(std::shared_ptr<sul_base>& sul) : eq_oracle_base(sul), use_sinks(USE_SINKS) {
        //search_strategy = std::unique_ptr<search_base>(new fringe_walk(MAX_CEX_LENGTH));
        search_strategy = std::unique_ptr<search_base>(new random_w_method(MAX_CEX_LENGTH));
        //search_strategy = std::unique_ptr<search_base>(new targeted_bfs_walk(MAX_CEX_LENGTH));
        // search_strategy = std::unique_ptr<search_base>(new bfs_strategy(8)); // number here is maximum length of
        // sequence. Find a better way to set this
        assert(dynamic_cast<input_file_sul*>(sul.get()) == nullptr);

        merger = nullptr;
    };

    std::optional<std::pair<std::vector<int>, int>>
    equivalence_query(state_merger* merger, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher);

    void initialize(state_merger* merger) override;
};

#endif
