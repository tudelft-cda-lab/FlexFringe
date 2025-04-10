#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include "parameters.h"
#include "paul_heuristic_2.h"
#include "input/inputdatalocator.h"

#include <map>
#include <set>
#include <unordered_set>

#include <cmath>
#include <iostream>

REGISTER_DEF_TYPE(paul_heuristic_2);
REGISTER_DEF_DATATYPE(paul_data_2);

void paul_data_2::print_state_label(std::iostream& output){
    count_data::print_state_label(output);
/*     output << "\nc : " << std::to_string(lm_confidence) << " | nm : " << std::to_string(num_merged) << "\n";

    for(auto c: all_confidences){
        output << std::to_string(c.first) << " | " << std::to_string(c.second) << "\n";
    } */
};

void paul_data_2::add_tail(tail* t) {
    // protects against multiple types that can result from distinguishing sequences
    if(final_counts.size() > 0)
        return;

    count_data::add_tail(t);
}

void paul_data_2::update(evaluation_data* right){
    auto r_data = static_cast<paul_data_2*>(right);

    for(auto c: r_data->all_confidences){
        if(c.first == -1) continue;

        float q_c = map_confidence(c.first);
        if(all_confidences.contains(q_c))
            all_confidences[q_c] += c.second;
        else
            all_confidences[q_c] = c.second;
    }

    if(!this->label_is_queried()){
        return;
    }
    else if(!r_data->label_is_queried()){ // the right one is from real data, while left (this) is not

        if(num_merged==0){
            final_counts_backup = final_counts; // do a copy
            final_counts.clear();
        }

        for(auto & final_count : r_data->final_counts){
            int type = final_count.first;
            int count = final_count.second;
            if(final_counts.contains(type)){
                final_counts[type] += count;
            } else {
                final_counts[type] = count;
            }
        }

        for(auto & path_count : r_data->path_counts){
            int type = path_count.first;
            int count = path_count.second;
            if(path_counts.contains(type)){
                path_counts[type] += count;
            } else {
                path_counts[type] = count;
            }
        }

        total_paths += r_data->total_paths;
        total_final += r_data->total_final;

    }
    else{ // both nodes are inferred by the transformer
        for(auto & final_count : r_data->final_counts){
            int type = final_count.first;
            int count = final_count.second;
            if(final_counts.contains(type)){
                final_counts[type] += count;
            } else {
                final_counts[type] = count;
            }
        }

        for(auto & path_count : r_data->path_counts){
            int type = path_count.first;
            int count = path_count.second;
            if(path_counts.contains(type)){
                path_counts[type] += count;
            } else {
                path_counts[type] = count;
            }
        }
        
        total_paths += r_data->total_paths;
        total_final += r_data->total_final;
    }

    num_merged++;
}



void paul_data_2::undo(evaluation_data* right){
    auto r_data = static_cast<paul_data_2*>(right);
    //this->lm_confidence -= r_data->lm_confidence;
    
    for(auto c: r_data->all_confidences){
        if(c.first == -1) continue;

        float q_c = map_confidence(c.first);
        all_confidences[q_c] -= c.second;
    }

    num_merged--;
    if(!this->label_is_queried()){
        return;
    }

    else if(!r_data->label_is_queried()){ // the right one is from real data, while left (this) is inferred
        if(num_merged == 0){
            final_counts = final_counts_backup;
            return;
        }

        for(auto & final_count : r_data->final_counts){
            int type = final_count.first;
            int count = final_count.second;
            final_counts[type] -= count;
        }
        for(auto & path_count : r_data->path_counts){
            int type = path_count.first;
            int count = path_count.second;
            path_counts[type] -= count;
        }
        total_paths -= r_data->total_paths;
        total_final -= r_data->total_final;
    }
    else{ // both are inferred
        for(auto & final_count : r_data->final_counts){
            int type = final_count.first;
            int count = final_count.second;
            final_counts[type] -= count;
        }
        for(auto & path_count : r_data->path_counts){
            int type = path_count.first;
            int count = path_count.second;
            path_counts[type] -= count;
        }
        total_paths -= r_data->total_paths;
        total_final -= r_data->total_final;
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
int paul_heuristic_2::check_for_consistency(paul_data_2* left, paul_data_2* right, int mismatch_count) const {
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

bool paul_heuristic_2::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    if(inconsistency_found) return false;    
    //if(!TYPE_CONSISTENT) return true;    
    auto* l = (paul_data_2*)left->get_data();
    auto* r = (paul_data_2*)right->get_data();  

/*     if(l->label_is_queried() && r->label_is_queried()){
        n_inferred_inferred_mismatches = check_for_consistency(l, r, n_inferred_inferred_mismatches);
        ++n_inferred_inferred_pairs;

        return true;
    }
    else if(l->label_is_queried() || r->label_is_queried()){
        n_real_inferred_mismatches = check_for_consistency(l, r, n_real_inferred_mismatches);
        ++n_real_inferred_pairs; 

        return true;
    } */

    // real_real must match, else inconsistent
    if(check_for_consistency(l, r) > 0){
        inconsistency_found = true;
        return false;
    }

    auto l_distribution = ii_handler->predict_node_with_sul(*(merger->get_aut()), left);
    auto r_distribution = ii_handler->predict_node_with_sul(*(merger->get_aut()), right);
    if(!ii_handler->distributions_consistent(l_distribution, r_distribution)){
        std::cout << " dx ";
        std::cout.flush();
        return false;
    }
        
    ++n_real_real_pairs;
    return true;
}

bool paul_heuristic_2::overlap_consistent(apta_node* left, apta_node* right) const {
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



double paul_heuristic_2::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    //if(!overlap_consistent(left, right)) return -1;

    return static_cast<double>(n_real_real_pairs)/*  + 0.1 * n_real_inferred_pairs + 0.05 * n_inferred_inferred_pairs */;
};

void paul_heuristic_2::reset(state_merger *merger){
    count_driven::reset(merger);

    n_inferred_inferred_pairs=0;
    n_inferred_inferred_mismatches=0;
    
    n_real_inferred_pairs=0;
    n_real_inferred_mismatches=0;
    
    n_real_real_pairs=0;
};

void paul_heuristic_2::provide_ii_handler(std::shared_ptr<ii_base>& ii_handler){
    this->ii_handler=ii_handler;
}

