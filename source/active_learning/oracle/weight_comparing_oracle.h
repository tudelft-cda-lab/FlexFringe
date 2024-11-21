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

/* #include "oracle_base.h"
#include "state_merger.h"
#include "parameters.h"

#include <optional>
#include <utility>

class weight_comparing_oracle : public oracle_base {
  private:
    const bool use_sinks; 

  protected:
    virtual void reset_sul() override{};

    bool
    test_trace_accepted(apta& hypothesis, trace* const tr, [[maybe_unused]] const std::unique_ptr<oracle_base>& oracle, inputdata& id);

  public:
    weight_comparing_oracle(const std::unique_ptr<sul_base>& sul) : oracle_base(sul), use_sinks(USE_SINKS) {
        conflict_detector = std::make_unique<type_conflict_detector>(sul);
        conflict_searcher = std::make_unique<linear_conflict_search>(conflict_detector);
    };

    std::optional<std::pair<std::vector<int>, sul_response>>
    equivalence_query(state_merger* merger) override;
}; */

#endif
