/**
 * @file paul_oracle.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _PAUL_ORACLE_H_
#define _PAUL_ORACLE_H_

#include "base_oracle.h"
#include "parameters.h"

#include "type_overlap_conflict_detector.h"
#include "suffix_tree.h"

#include <optional>
#include <utility>

class paul_oracle : public base_oracle {
  private:
    struct resp_t final {
      private:
        int pred_type;
        double confidence;

      public:
        explicit resp_t(int&& pred_type, double&& confidence){
          this->pred_type = pred_type;
          this->confidence = confidence;
        }

        resp_t() = delete;

        int get_type() const noexcept {
          return pred_type;
        }

        double get_confidence() const noexcept {
          return confidence;
        }
    };

    inline resp_t get_sul_response(const std::vector< std::vector<int> >& query_string, inputdata& id) const;
    inline bool check_test_string_interesting(const double confidence) const noexcept;

  public:
    paul_oracle(const std::shared_ptr<sul_base>& sul, const std::shared_ptr<ii_base>& ii_handler) : base_oracle(sul) {
      if(!ii_handler)
        throw std::invalid_argument("ERROR: ii_handler not provided to paul oracle, but it depends on it.");

      //conflict_detector = conflict_detector_factory::create_detector(sul); // TODO: perhaps use the one with the ii_handler?
      //conflict_searcher = std::make_unique<linear_conflict_search>(conflict_detector);
      
      tested_strings = std::make_unique<suffix_tree>();
    };

    std::unique_ptr<suffix_tree> tested_strings;

    std::optional<std::pair<std::vector<int>, sul_response>>
    equivalence_query(state_merger* merger) override;
};

#endif
