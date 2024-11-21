/**
 * @file string_probability_conflict_detector.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "string_probability_conflict_detector.h"
#include "apta.h"
#include "probabilistic_heuristic_interface.h"

#include "parameters.h"
#include "common_functions.h"

#include <limits>

using namespace std;

/**
 * @brief Parses hypothesis with string (substr), returns the parsed probability assigned to the string.
 */
const float string_probability_conflict_detector::get_string_prob(const std::vector<int>& substr, apta& hypothesis, inputdata& id) const {
  static const bool use_sinks = USE_SINKS; 
  apta_node* n = hypothesis.get_root();
  tail* t = test_tr->get_head();

  vector<int> current_substring;
  double sampled_probability = 1;
  while (t != nullptr) {
    if (use_sinks && n==nullptr) {
      cout << "Sink found in eq" << endl;
      return 0; // TODO: does this make sense?
    }
    else if (n == nullptr) {
      cout << "Conflict because tree not parsable" << endl;
      return numeric_limits<float>::min();; // ensures that condition will not be satisfied (uness there's a very extreme case which is unlikely to happen)
    }

    if (t->is_final() && FINAL_PROBABILITIES) {
      sampled_probability *= static_cast<probabilistic_heuristic_interface_data*>(n->get_data())->get_final_probability(); // TODO: predict probability better here
      break;
    } else if (t->is_final())
        break;

    const int symbol = t->get_symbol();
    sampled_probability *= static_cast<probabilistic_heuristic_interface_data*>(n->get_data())->get_normalized_probability(symbol);

    n = active_learning_namespace::get_child_node(n, t);
    t = t->future();
  }

  return sampled_probability;
}

/**
 * @brief Compares the parsed string through the hypothesis with the prediction. If larger than a bound mu, then we raise an alarm.
 */
pair<bool, optional<sul_response> > string_probability_conflict_detector::creates_conflict(const vector<int>& substr, apta& hypothesis, inputdata& id) {
  static const float mu = MU;
  
  sul_response resp = sul->do_query(substr, id);
  float sampled_probability = get_string_prob(substr, hypothesis, id);
  if(sampled_probability == -1)
    return make_pair(true, resp);
  
  double true_val = resp.GET_DOUBLE();
  float diff = abs(true_val - sampled_probability);
  if (diff > mu) {
      cout << "Predictions of the following counterexample: The true probability: " << true_val
           << ", predicted probability: " << sampled_probability << endl;

      return make_pair(true, resp);
  }

  return make_pair(false, nullopt);
}