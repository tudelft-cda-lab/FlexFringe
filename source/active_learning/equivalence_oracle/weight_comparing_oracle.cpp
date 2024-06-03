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
#include "parameters.h"
#include "weight_comparator.h"

#include <cmath>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Returns true if the test trace is accepted, else 0.
 */
bool
weight_comparing_oracle::test_trace_accepted(apta& hypothesis, trace* const tr, [[maybe_unused]] const std::unique_ptr<base_teacher>& teacher, inputdata& id){
    static const auto mu = static_cast<double>(MU);
    static const auto EOS = END_SYMBOL;

    apta_node* n = hypothesis.get_root();
    tail* t = tr->get_head();

    vector<int> current_substring;
    double sampled_value = 1;
    double true_value = 1;

    while (t != nullptr) {
        if (n == nullptr) {
            cout << "Counterexample because tree not parsable" << endl;
            return false;
        }

        auto weights = teacher->get_weigth_distribution(current_substring, id);
        if (t->is_final() && EOS != -1) {
            sampled_value *= static_cast<weight_comparator_data*>(n->get_data())->get_final_weight();
            true_value *= weights[EOS];
            break;
        } else if (t->is_final())
            break;

        const int symbol = t->get_symbol();
        sampled_value *= static_cast<weight_comparator_data*>(n->get_data())->get_weight(symbol);
        const int mapped_symbol = stoi(id.get_symbol(symbol));
        true_value *= weights[mapped_symbol];

        current_substring.push_back(symbol);
        n = active_learning_namespace::get_child_node(n, t);
        t = t->future();
    }

    auto diff = abs(true_value - sampled_value);
    if (diff > mu) {
        cout << "Predictions of the following counterexample: The true probability: " << true_value
            << ", predicted probability: " << sampled_value << endl;
        return false;
    }

    return true;
}

/**
 * @brief This function does what you think it does. The type does not matter, because we perform this one
 * on SULs that return us probabilities, hence it only returns a dummy value.
 *
 * Note: Requires that SOS and EOS be there.
 *
 * @param merger The merger.
 * @param teacher The teacher.
 * @return std::optional< pair< vector<int>, int> > nullopt if no counterexample found, else the counterexample. The
 * type (integer) will always be zero.
 */
std::optional<pair<vector<int>, int>>
weight_comparing_oracle::equivalence_query(state_merger* merger, const unique_ptr<base_teacher>& teacher) {
    inputdata& id = *(merger->get_dat());
    apta& hypothesis = *(merger->get_aut());

    static const auto& alphabet = id.get_alphabet();
    static const auto mu = static_cast<double>(MU);
    static const auto EOS = END_SYMBOL;

    static const bool CHECK_ACCESS_STRINGS = true;
    
    if(CHECK_ACCESS_STRINGS){
        cout << "Testing the access strings" << endl;
        for (red_state_iterator r_it = red_state_iterator(hypothesis.get_root()); *r_it != nullptr; ++r_it) {
            auto n = *r_it;
            while(n!=nullptr){
                trace* at = n->get_access_trace();
                if(!test_trace_accepted(hypothesis, at, teacher, id)){
                    vector<int> query_string = at->get_input_sequence(true, false);
                    return make_optional<pair<vector<int>, int>>(make_pair(query_string, 0));
                }
                n = n->get_next_merged();
            }
        }
    }

    // here we start the actual search
    search_strategy->reset();
    std::optional<vector<int>> query_string_opt = search_strategy->next(id);
    while (query_string_opt != nullopt) {
        auto& query_string = query_string_opt.value();
        trace* test_tr = vector_to_trace(query_string, id, 0); // type-argument irrelevant here

        if(!test_trace_accepted(hypothesis, test_tr, teacher, id))
            return make_optional<pair<vector<int>, int>>(make_pair(query_string, 0));

        query_string_opt = search_strategy->next(id);
    }

    return nullopt;
}

/**
 * @brief Initializes the search strategy. Needed for e.g. W-method.
 * 
 * @param merger The merger.
 */
void weight_comparing_oracle::initialize(state_merger* merger){
    this->search_strategy->initialize(merger);
}