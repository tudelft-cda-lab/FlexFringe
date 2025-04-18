/**
 * @file conflict_detector_factory.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-04-18
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "conflict_detector_factory.h"

#include "type_conflict_detector.h"
#include "string_probability_conflict_detector.h"
#include "type_overlap_conflict_detector.h"

#include <unordered_set>
#include <string_view>

using namespace std;

namespace {
  static const unordered_set<string_view> type_based_algorithms{
    "lsharp",
    "lstar",
    "ldot",
    "lstar_imat",
    "paul"
  };

  static const unordered_set<string_view> string_probability_based_algorithms{
    "probabilistic_lsharp",
    "weighted_lsharp"
  };

  static const unordered_set<string_view> type_overlap_based_algorithms{
    // perhaps PAUL?
  };
};

/**
 * @brief Does what you think it does. 
 * Since the conflict detector is tightly knitted to the kind of algorithm 
 * that is used, the selection is based on this.
 */
shared_ptr<conflict_detector_base> conflict_detector_factory::create_detector(const std::shared_ptr<sul_base>& sul) {
  if(type_based_algorithms.contains(ACTIVE_LEARNING_ALGORITHM))
    return make_shared<type_conflict_detector>(sul);
  else if(string_probability_based_algorithms.contains(ACTIVE_LEARNING_ALGORITHM))
    return make_shared<string_probability_conflict_detector>(sul); 
  else
    throw invalid_argument("No conflict detector is registered with this algorithm. Please choose for this.");
}

/**
 * @brief Does what you think it does. 
 * Since the conflict detector is tightly knitted to the kind of algorithm 
 * that is used, the selection is based on this.
 */
shared_ptr<conflict_detector_base> conflict_detector_factory::create_detector(const std::shared_ptr<sul_base>& sul, const std::shared_ptr<ii_base>& ii_handler) {
  if(type_overlap_based_algorithms.contains(ACTIVE_LEARNING_ALGORITHM))
    return make_shared<type_overlap_conflict_detector>(sul, ii_handler);
  else
    throw invalid_argument("No conflict detector is registered with this algorithm. Please choose for this.");
}