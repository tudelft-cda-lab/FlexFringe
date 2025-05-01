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
 * @brief Go through the hypothesis step by step, compute the probability step by step for both the hypothesis and teh SUL simultaneously, and raise a conflict if distributions clash. 
 * Elsewise, if end of string reached with no clash, then no conflict detected.
 */
pair<bool, optional<sul_response> > string_probability_conflict_detector::creates_conflict_compute_step_by_step(const vector<int>& substr, apta& hypothesis, inputdata& id, const bool sul_requires_batch_input){
  static const bool use_sinks = USE_SINKS;
  static const float mu = MU;

  static const auto ask_sul_lambda = [](const vector<int>& substr, const shared_ptr<sul_base>& sul, inputdata& id, const bool sul_requires_batch_input) -> vector<double> {
    if(sul_requires_batch_input){
      vector< vector<int> > query_batch;
      query_batch.push_back(substr);
      return sul->do_query(query_batch, id).GET_DOUBLE_VEC(); // assumes we get a weight distribution obviously
    }
    return sul->do_query(substr, id).GET_DOUBLE_VEC();
  };
  
  vector<int> current_substring;
  double sampled_probability = 1;
  double true_probability = 1;

  apta_node* n = hypothesis.get_root();
  for (const int symbol : substr) {
    const auto weights = ask_sul_lambda(current_substring, sul, id, sul_requires_batch_input);

    if (use_sinks && n==nullptr) {
      cout << "Sink found in eq" << endl;

      true_probability *= weights[AL_END_SYMBOL]; // TODO: bake the EOS index into the Python script, along with the other requirements
        
      // assume: sampled_probability = 0
      if(true_probability < mu) 
        return make_pair(false, nullopt);
      else{
        cout << "Conflict because we hit sink state and probability does not align" << endl;
        return make_pair(true, make_optional(sul_response(true_probability)));
      }
    }
    else if (n == nullptr) {
      cout << "Conflict because tree not parsable" << endl;
      return make_pair(true, make_optional(sul_response(true_probability)));
    }

    const int mapped_symbol = stoi(id.get_symbol(symbol));
    true_probability *= weights[mapped_symbol];
    sampled_probability *= static_cast<probabilistic_heuristic_interface_data*>(n->get_data())->get_probability(symbol);

    n = n->get_child(symbol);
    current_substring.push_back(symbol);
  }

  if (FINAL_PROBABILITIES){
    const auto weights = ask_sul_lambda(current_substring, sul, id, sul_requires_batch_input);
    sampled_probability *= static_cast<probabilistic_heuristic_interface_data*>(n->get_data())->get_final_probability(); // TODO: predict probability better here
    true_probability *= weights[AL_END_SYMBOL];
  }

  auto diff = abs(true_probability - sampled_probability);
  if (diff > mu) {
      cout << "Predictions of the following counterexample: The true probability: " << true_probability
          << ", predicted probability: " << sampled_probability << endl;
      return make_pair(true, make_optional(sul_response(true_probability)));;
  }
  return make_pair(false, nullopt);
}

/**
 * @brief Parses hypothesis with string (substr), returns the parsed probability assigned to the string.
 */
const float string_probability_conflict_detector::get_string_prob_from_hypothesis(const vector<int>& substr, apta& hypothesis, inputdata& id) const {
  static const bool use_sinks = USE_SINKS; 
  apta_node* n = hypothesis.get_root();
  //tail* t = test_tr->get_head();

  vector<int> current_substring;
  double sampled_probability = 1;
  //while (t != nullptr) {
  for (const int symbol : substr) {
    if (use_sinks && n==nullptr) {
      cout << "Sink found in eq" << endl;
      return 0; // TODO: does this make sense?
    }
    else if (n == nullptr) {
      cout << "Conflict because tree not parsable" << endl;
      return numeric_limits<float>::min();; // ensures that condition will not be satisfied (uness there's a very extreme case which is unlikely to happen)
    }

    //if (FINAL_PROBABILITIES) {
    //  sampled_probability *= static_cast<probabilistic_heuristic_interface_data*>(n->get_data())->get_final_probability(); // TODO: predict probability better here
      //break;
    //} else if (t->is_final())
    //    break;

    //const int symbol = t->get_symbol();
    sampled_probability *= static_cast<probabilistic_heuristic_interface_data*>(n->get_data())->get_probability(symbol);

    n = n->get_child(symbol);
    //t = t->future();
  }

  if (FINAL_PROBABILITIES)
    sampled_probability *= static_cast<probabilistic_heuristic_interface_data*>(n->get_data())->get_final_probability(); // TODO: predict probability better here

  return sampled_probability;
}

/**
 * @brief Checks for conflict on the string probability. Assumes that resp already has the full computed string-probability of substr as given by the SUL.
 * Else, use creates_conflict_compute_step_by_step() instead. 
 */
pair<bool, optional<sul_response> > string_probability_conflict_detector::creates_conflict_common(const sul_response& resp, const vector<int>& substr, apta& hypothesis, inputdata& id){
  static const float mu = MU;
  
  float sampled_probability = get_string_prob_from_hypothesis(substr, hypothesis, id);
  if(sampled_probability == -1)
    return make_pair(true, resp);
  
  double true_val = resp.has_double_val() ? resp.GET_DOUBLE() : resp.GET_DOUBLE_VEC().at(0);

  float diff = abs(true_val - sampled_probability);
  if (diff > mu) {
      cout << "Predictions of the following counterexample: The true probability: " << true_val
           << ", predicted probability: " << sampled_probability << endl;

      return make_pair(true, resp);
  }

  return make_pair(false, nullopt);
}

/**
 * @brief Compares the parsed string through the hypothesis with the prediction. If larger than a bound mu, then we raise an alarm.
 */
pair<bool, optional<sul_response> > string_probability_conflict_detector::creates_conflict(const vector< vector<int> >& substr, apta& hypothesis, inputdata& id) {
  if(SUL_OUTPUTS_STRING_PROBABILITY){
    sul_response resp = sul->do_query(substr, id);
    return creates_conflict_common(resp, substr.at(0), hypothesis, id);
  }
  return creates_conflict_compute_step_by_step(substr.at(0), hypothesis, id, true);
}

/**
 * @brief Compares the parsed string through the hypothesis with the prediction. If larger than a bound mu, then we raise an alarm.
 */
pair<bool, optional<sul_response> > string_probability_conflict_detector::creates_conflict(const vector<int>& substr, apta& hypothesis, inputdata& id) {
  if(SUL_OUTPUTS_STRING_PROBABILITY){
    sul_response resp = sul->do_query(substr, id);
    return creates_conflict_common(resp, substr, hypothesis, id);
  }
  return creates_conflict_compute_step_by_step(substr, hypothesis, id, false);
}