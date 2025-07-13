#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include "parameters.h"
#include "paul_heuristic.h"
#include "input/inputdatalocator.h"

#include <map>
#include <set>
#include <unordered_set>

#include <cmath>
#include <iostream>

#ifdef __FLEXFRINGE_CUDA
#include "source/active_learning/active_learning_util/cuda/cuda_common.cuh"
#include <ranges>
#endif

REGISTER_DEF_TYPE(paul_heuristic);
REGISTER_DEF_DATATYPE(paul_data);

void paul_data::print_state_label(std::iostream& output){
    count_data::print_state_label(output);
};

void paul_data::add_tail(tail* t) {
    if(t->get_type() != -1) // making sure we don't add unlabeled traces
        count_data::add_tail(t);
}

/**
 * @brief Two ways. If we have real labeled data, e.g. from the training data, then we 
 * use that information. If not, we check for the inferred type. If both fail we omit this one.
 */
int paul_data::predict_type(tail* t) {
    if(final_counts.size()>0)
        return count_data::predict_type(t);
    
    // predict the maximum
    int res = 0;
    double max_count = -1;
    for(int i = 0; i < inputdata_locator::get()->get_types_size(); ++i){ // -1 is the unknown type
        double prob = predict_type_score(i);
        if(max_count == -1 || max_count < prob){
            max_count = prob;
            res = i;
        }
    }
    return res;
}

void paul_data::update(evaluation_data* right){
    count_data::update(right);

    auto r_data = static_cast<paul_data*>(right);
    for(auto const& [type, count]: r_data->inferred_final_counts){
        inferred_final_counts[type] += count;
        inferred_total_final += count;
    }
}

/**
 * @brief Overwrites the current set of predictions with the ones handed to this method.
 */
void paul_data::set_predictions(layer_predictions_map&& predictions){
#ifdef __FLEXFRINGE_CUDA

#ifndef gpuErrcheck
#define gpuErrchk(ans) { cuda_common::gpuAssert((ans), __FILE__, __LINE__); }

    std::unordered_map< int, int* > preds_d(predictions.size());
    std::unordered_map< int, size_t > sizes(predictions.size());

    for(const auto& [len, preds]: predictions){
        sizes[len] = preds.size();
        
        preds_d[len] = nullptr; // making sure the entry
        int* target_field_d = preds_d[len];
        const auto byte_size = preds.size() * sizeof(int);
        gpuErrchk(cudaMalloc((void**) &target_field_d, byte_size));
        gpuErrchk(cudaMemcpy(target_field_d, preds.data(), byte_size, cudaMemcpyHostToDevice));
    }
    
    this->predictions.len_pred_map_d = std::move(preds_d);
    this->predictions.len_size_map = std::move(sizes);
#endif // gpuErrcheck

#else
    this->predictions = move(predictions);
#endif // __FLEXFRINGE_CUDA
}

#ifdef __FLEXFRINGE_CUDA
paul_data::~paul_data(){
#ifndef gpuErrcheck
#define gpuErrchk(ans) { cuda_common::gpuAssert((ans), __FILE__, __LINE__); }

    for(int* preds_d: this->predictions.len_pred_map_d | std::ranges::views::values){
       gpuErrchk(cudaFree(preds_d));
    }
}
#endif // gpuErrcheck
#endif // __FLEXFRINGE_CUDA
/**
 * @brief Does what you think it does.
 */
void paul_data::set_confidence(const float confidence) noexcept { 
  lm_confidence = confidence;
};

/**
 * @brief Does what you think it does.
 */
void paul_data::add_inferred_type(const int t) noexcept{
    inferred_final_counts[t]++;
    inferred_total_final++;
}

void paul_data::undo(evaluation_data* right){
    count_data::undo(right);

    auto r_data = static_cast<paul_data*>(right);
    for(auto const& [type, count]: r_data->inferred_final_counts){
        inferred_final_counts[type] -= count;
        inferred_total_final -= count;
    }
}

/**
 * @brief Compares the two nodes, and if mismatch found returns mismatch_count+1, else mismatch_count.
 * 
 * @param left left data
 * @param right right data
 * @param mismatch_count Current count of mismatches. Default=0.
 * @return int If mismatch found returns mismatch_count+1, else mismatch_count
 */
int paul_heuristic::check_for_consistency(paul_data* left, paul_data* right, int mismatch_count) const {
    for(auto & final_count : left->final_counts){
        int type = final_count.first;
        int count = final_count.second;
        if(count != 0){
            for(auto & final_count2 : right->final_counts){
                int type2 = final_count2.first;
                int count2 = final_count2.second;
                if(count2 != 0 && type2 != type){
                    return mismatch_count+1;
                }
            }
        }
    }
    return mismatch_count;
}

bool paul_heuristic::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    if(inconsistency_found) return false;    
    //if(!TYPE_CONSISTENT) return true;    
    auto* l = (paul_data*)left->get_data();
    auto* r = (paul_data*)right->get_data();  

    // real_real must match, else inconsistent
    if(check_for_consistency(l, r) > 0){
        inconsistency_found = true;
        return false;
    }

/*     const static int MAX_C_DEPTH = -1;
    if(MAX_C_DEPTH < 0 || depth <= MAX_C_DEPTH){
        auto l_distribution = ii_handler->predict_node_with_sul(*(merger->get_aut()), left);
        auto r_distribution = ii_handler->predict_node_with_sul(*(merger->get_aut()), right);
        if(!ii_handler->distributions_consistent(l_distribution, r_distribution)){
            std::cout << " dx ";
            std::cout.flush();
            inconsistency_found = true;
            return false;
        }
    } */

    ++n_real_real_pairs;
    return true;
}

bool paul_heuristic::overlap_consistent(apta_node* left, apta_node* right) const {
#ifndef NDEBUG
    //cout << "\nl: " << left->get_number() << ", depth: " << left->get_depth() << "|r: " << right->get_number() << ", depth: " << right->get_depth() << "\n";
    //cout << "n_real_pairs: " << n_real_real_pairs << ", n_real_inf_pairs: " << n_real_inferred_pairs << ", n_inf_inf_pairs: " << n_inferred_inferred_pairs << "\n";
    //cout << "n_real_inf_mismatches: " << n_real_inferred_mismatches << ", n_inf_inf_mismatches: " << n_inferred_inferred_mismatches << "\n" << endl;
#endif

    static const float alpha = CHECK_PARAMETER;

    float l1 = static_cast<float>(n_real_inferred_mismatches) / static_cast<float>(n_real_inferred_pairs);
    float l2 = static_cast<float>(n_inferred_inferred_mismatches) / static_cast<float>(n_inferred_inferred_pairs);

    float r1 = sqrt(0.5 * log(2 / alpha)) * (1 / sqrt(n_real_inferred_pairs));
    float r2 = sqrt(0.5 * log(2 / alpha)) * (1 / sqrt(n_inferred_inferred_pairs));

    r1 = std::max(r1, float(0.01));
    r2 = std::max(r2, float(0.01));

    if(l1 > r1 || l2 > r2)
        return false;

    return true;
   

/*     double n_inf_sum = n_inferred_inferred_pairs + n_real_inferred_pairs;
    double sum = n_real_real_pairs + n_inf_sum;

    if(double(n_real_inferred_mismatches + n_inferred_inferred_mismatches) / n_inf_sum > 0.01) // more than [threshold] percent wrong -> think about this one
        return false;

    return true; */
}



double paul_heuristic::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    //if(!overlap_consistent(left, right)) return -1;

    return static_cast<double>(n_real_real_pairs)/*  + 0.1 * n_real_inferred_pairs + 0.05 * n_inferred_inferred_pairs */;
};

void paul_heuristic::reset(state_merger *merger){
    count_driven::reset(merger);

    n_inferred_inferred_pairs=0;
    n_inferred_inferred_mismatches=0;
    
    n_real_inferred_pairs=0;
    n_real_inferred_mismatches=0;
    
    n_real_real_pairs=0;
};

void paul_heuristic::provide_ds_handler(std::shared_ptr<distinguishing_sequences_handler_base>& ii_handler){
    this->ii_handler=ii_handler;
}

