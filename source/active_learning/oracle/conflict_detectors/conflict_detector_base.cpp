/**
 * @file conflict_detector_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-04-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "conflict_detector_base.h"

using namespace std;

/**
 * @brief Compares the parsed string through the hypothesis with the prediction. If larger than a bound mu, then we raise an alarm.
 */
pair<bool, optional<sul_response> > conflict_detector_base::creates_conflict(const vector< vector<int> >& substr, apta& hypothesis, inputdata& id) {
  sul_response resp = sul->do_query(substr, id);
  return creates_conflict_common(resp, substr.at(0), hypothesis, id);
}

/**
 * @brief Compares the parsed string through the hypothesis with the prediction. If larger than a bound mu, then we raise an alarm.
 */
pair<bool, optional<sul_response> > conflict_detector_base::creates_conflict(const vector<int>& substr, apta& hypothesis, inputdata& id) {
  sul_response resp = sul->do_query(substr, id);
  return creates_conflict_common(resp, substr, hypothesis, id);
}