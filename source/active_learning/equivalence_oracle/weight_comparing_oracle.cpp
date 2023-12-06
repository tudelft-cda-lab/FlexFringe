/**
 * @file weight_comparing_oracle.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "weight_comparing_oracle.h"
#include "common_functions.h"
#include "weight_comparator.h"
#include "parameters.h"

#include <cmath>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief This function does what you think it does. The type does not matter, because we perform this one 
 * on SULs that return us probabilities, hence it only returns a dummy value.
 * 
 * Note: Requires that SOS and EOS be there.
 * 
 * @param merger The merger.
 * @param teacher The teacher.
 * @return std::optional< pair< vector<int>, int> > nullopt if no counterexample found, else the counterexample. The type (integer) will 
 * always be zero.
 */
std::optional< pair< vector<int>, int> > weight_comparing_oracle::equivalence_query(state_merger* merger, const unique_ptr<base_teacher>& teacher) {  
  inputdata& id = *(merger->get_dat());
  apta& hypothesis = *(merger->get_aut());

  static const auto& alphabet = id.get_alphabet(); 
  static const auto mu = static_cast<double>(MU);
  static const auto EOS = END_SYMBOL;

  std::optional< vector<int> > query_string_opt = search_strategy->next(id);
  while(query_string_opt != nullopt){
    auto& query_string = query_string_opt.value();

    trace* test_tr = vector_to_trace(query_string, id, 0); // type-argument irrelevant here

    apta_node* n = hypothesis.get_root();
    tail* t = test_tr->get_head();

    vector<int> current_substring;
    
    double sampled_value = 1;
    double true_value = 1;
    while(t!=nullptr){        
        if(n == nullptr){
          cout << "Counterexample because tree not parsable" << endl;
          search_strategy->reset();
          return make_optional< pair< vector<int>, int > >(make_pair(query_string, 0));
        }

        auto weights = teacher->get_weigth_distribution(current_substring, id);
        if(t->is_final() && EOS != -1){
          sampled_value *= static_cast<weight_comparator_data*>(n->get_data())->get_final_weight();
          true_value *= weights[EOS];
          break;
        }
        else if(t->is_final())
          break;

        const int symbol = t->get_symbol();
        sampled_value *= static_cast<weight_comparator_data*>(n->get_data())->get_weight(symbol);
        true_value *= weights[symbol];
        
        current_substring.push_back(symbol);
        n = active_learning_namespace::get_child_node(n, t);
        t = t->future();
    }

    auto diff = abs(true_value - sampled_value);
    if(diff > mu){
      cout << "Predictions of the following counterexample: The true probability: " << true_value << ", predicted probability: " << sampled_value << endl;
      search_strategy->reset();
      return make_optional< pair< vector<int>, int > >(make_pair(query_string, 0));
    } 

    query_string_opt = search_strategy->next(id);
  }

  return nullopt;
}
