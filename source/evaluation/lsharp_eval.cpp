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


/* default evaluation, count number of performed merges */
bool lsharp_eval::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    if(inconsistency_found) return false;    
    if(!TYPE_CONSISTENT) return true;    
    auto* l = (count_data*)left->get_data();
    auto* r = (count_data*)right->get_data();

    for(auto & final_count : l->final_counts){
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
    
    return true;
};