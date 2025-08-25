#include "state_merger.h"
#include "lsharp_eval.h"
#include "evaluation_factory.h"
#include "count_types.h"
#include "input/inputdatalocator.h"

#include <map>
#include <set>
#include <unordered_set>

REGISTER_DEF_TYPE(lsharp_eval);
REGISTER_DEF_DATATYPE(lsharp_data);

/**
 * @brief Compares the two maps, and if the types match return true, else false.
 */
bool lsharp_eval::types_match(const std::unordered_map<int, int>& m1, const std::unordered_map<int, int>& m2) const noexcept {
    for(const auto& counts : m1){
        const int type = counts.first;
        const int count = counts.second;
        
        if(count != 0){
            for(auto& counts2 : m2){
                const int type2 = counts2.first;
                const int count2 = counts2.second;
                
                if(count2 != 0 && type2 != type)
                    return false;
            }
        }
    }

    return true;
}

/**
 * @brief Checks the final types (the types of the nodes), and merges only if those are consistent.
 */
bool lsharp_eval::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    if(inconsistency_found) return false;    
    //if(!TYPE_CONSISTENT) return true; // TODO: what to do with this one? 
    auto* l = (count_data*)left->get_data();
    auto* r = (count_data*)right->get_data();

    if(!types_match(r->final_counts, l->final_counts) || !types_match(l->final_counts, r->final_counts)){
        inconsistency_found = true;
        return false;
    }
    
    return true;
};

/** 
 * We might not have full information, but in case many merges are possible we 
 * want to merge into the lowest layer possible. Therefore, we use the difference in depth as well.
*/
double lsharp_eval::compute_score(state_merger* merger, apta_node* left, apta_node* right){
    double score_1 = count_driven::compute_score(merger, left, right);
    
    // left node is always the red one as per flexfringe convention
    auto left_depth = left->get_depth();
    auto right_depth = right->get_depth();
    double diff = static_cast<double>(abs(left_depth - right_depth));
    return score_1 + diff;
}
