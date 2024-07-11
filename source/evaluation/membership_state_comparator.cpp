/**
 * @file membership_state_comparator.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-06-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "state_merger.h"
#include "evaluate.h"
#include "membership_state_comparator.h"
#include "parameters.h"
#include "input/inputdatalocator.h"

#include <iostream>
#include <unordered_set>
#include <cmath>

REGISTER_DEF_DATATYPE(membership_state_comparator_data);
REGISTER_DEF_TYPE(membership_state_comparator);

void membership_state_comparator_data::print_state_label(iostream& output){
    evaluation_data::print_state_label(output);
    for(auto x: LS){
        output << x << "\n";
    }
};

/** Merging update and undo_merge routines */

void membership_state_comparator_data::update(evaluation_data* right){
    evaluation_data::update(right);

    auto other = static_cast<membership_state_comparator_data*>(right);
    N += other->N;
    for(int i = 0; i<LS.size(); ++i){
        LS[i] += other->LS[i];
        SS[i] += other->SS[i];
    }
};

void membership_state_comparator_data::undo(evaluation_data* right){
    evaluation_data::undo(right);

    auto other = static_cast<membership_state_comparator_data*>(right);
    N -= other->N;
    for(int i = 0; i<LS.size(); ++i){
        LS[i] -= other->LS[i];
        SS[i] -= other->SS[i];
    }
};

/**
 * @brief Updates the two functions we use for incrementatl statistics. We separated it from the 
 * actual computation to save overhead executions.
 * 
 */
void membership_state_comparator_data::update_sums(const std::vector<float>& internal_rep){
    N += 1;
    if(LS.size()==0){
        //cout << "Size of internal representation: " << internal_rep.size() << endl;
        LS = internal_rep;
        for(int i = 0; i < internal_rep.size(); ++i){
            SS.push_back(pow(internal_rep[i], 2));
        }
        if(LS.size() != SS.size()){
            cerr << "Error in evaluation function. LS and SS should have same length but have " << LS.size() << " and " << SS.size() << endl;
            throw exception();
        }

        // initializing here saves us resizing later on
        means = vector<float>(internal_rep.size());
        std_devs = vector<float>(internal_rep.size());

        return;
    }

    for(int i=0; i<internal_rep.size(); ++i){
        LS[i] += internal_rep[i];
        SS[i] += pow(internal_rep[i], 2);
    }
}

/**
 * @brief Triggers the computation of the statistics from LS to SS. We split up the 
 * two functions to over overhead computations.
 * 
 */
void membership_state_comparator_data::compute_statistics(){
    for(int i=0; i<LS.size(); ++i){
        means[i] = LS[i]/N;
        auto variance = abs(SS[i]/N - pow(means[i], 2));
        std_devs[i] = sqrt(SS[i]);
    }
}

float membership_state_comparator::get_diff(const vector<float>& left_v, const vector<float>& right_v) const {
    // we do the mean-square error
    float res = 0;
    for(int i = 0; i<left_v.size(); ++i){
        res += pow(left_v[i] - right_v[i], 2);
    }
    return sqrt(res);
}

/**
 * @brief TODO: add description of how it works
 */
bool membership_state_comparator::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    if(inconsistency_found) return false;    
    if(!TYPE_CONSISTENT) return true;

    static float mu = MU;

    auto* l = (membership_state_comparator_data*)left->get_data();
    auto* r = (membership_state_comparator_data*)right->get_data();

/*     l->compute_statistics();
    r->compute_statistics();

    auto mean_diff = get_diff(l->means, r->means);
    if(mean_diff > mu){
        inconsistency_found = true;
        return false;
    }
    cout << "mean_diff: " << mean_diff << endl;

    auto std_diff = get_diff(l->std_devs, r->std_devs);
    if(std_diff > mu){
        inconsistency_found = true;
        return false;
    }
    cout << "std_diff: " << std_diff << endl; */

    return lsharp_eval::consistent(merger, left, right, depth);

/*     for(auto & final_count : l->final_counts){
        int type = final_count.first;
        int count = final_count.second;
        if(count != 0){
            for(auto & final_count2 : r->final_counts){
                int type2 = final_count2.first;
                int count2 = final_count2.second;
                if(count2 != 0 && type2 != type){
                    inconsistency_found = true;
                    return false;
                }
            }
        }
    }

    for(auto & final_count : r->final_counts){
        int type = final_count.first;
        int count = final_count.second;
        if(count != 0){
            for(auto & final_count2 : l->final_counts){
                int type2 = final_count2.first;
                int count2 = final_count2.second;
                if(count2 != 0 && type2 != type){
                    inconsistency_found = true;
                    return false;
                }
            }
        }
    }
    
    return true; */
};

// double membership_state_comparator::compute_score(state_merger *merger, apta_node* left, apta_node* right){
//     return score;
// };

void membership_state_comparator::reset(state_merger *merger){
    inconsistency_found = false;
    score = 0;
};

// double membership_state_comparator::get_distance(apta* aut, apta_node* left_node, apta_node* right_node){
//     auto* l = static_cast<membership_state_comparator_data*>( left_node->get_data() );
//     auto* r = static_cast<membership_state_comparator_data*>( right_node->get_data() );
// 
//     auto right_sequence = right_node->get_access_trace()->get_input_sequence(true, false);
//     auto n = aut->get_root();
//     auto n_data = static_cast<membership_state_comparator_data*>( n->get_data() );
// 
//     float at_weight = 1;
//     for(auto symbol: right_sequence){
//         at_weight *= n_data->get_weight(symbol);
//         n = n->get_child(symbol);
//         n_data = static_cast<membership_state_comparator_data*>( n->get_data() );
//     }
// 
//     double diff = abs(at_weight * l->get_final_weight() - at_weight * r->get_final_weight());
//     return diff;
// };