#include "state_merger.h"
#include "evaluate.h"
#include <map>
#include "overlap.h"
#include "parameters.h"

REGISTER_DEF_DATATYPE(overlap_data);
REGISTER_DEF_TYPE(overlap_driven);

void overlap_data::print_transition_label(std::iostream& output, int symbol){
    output << pos(symbol) << " ";
};

void overlap_data::print_state_label(std::iostream& output){
    count_data::print_state_label(output);
    output << "\n" << num_paths() << " " << num_final();
};

/* Overlap driven, count overlap in positive transitions, used in Stamina winner */
bool overlap_driven::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    if(count_driven::consistent(merger, left, right, depth) == false){
        inconsistency_found = true;
        return false;
    }

    overlap_data* l = (overlap_data*) left->get_data();
    overlap_data* r = (overlap_data*) right->get_data();

    if(l->pos_paths() >= STATE_COUNT){
        for(num_map::iterator it = r->pos_begin(); it != r->pos_end(); ++it){
            if(it->second >= SYMBOL_COUNT && l->pos(it->first) == 0){
                inconsistency_found = true;
                return false;
            }
        }
    }

    if(r->pos_paths() >= STATE_COUNT){
        for(num_map::iterator it = l->pos_begin(); it != l->pos_end(); ++it){
            if(it->second >= SYMBOL_COUNT && r->pos(it->first) == 0){
                inconsistency_found = true;
                return false;
            }
        }
    }
    return true;
};

void overlap_driven::update_score(state_merger *merger, apta_node* left, apta_node* right){
    overlap_data* l = (overlap_data*) left->get_data();
    overlap_data* r = (overlap_data*) right->get_data();

    if (inconsistency_found) return;
    if (consistent(merger, left, right, 0) == false) return; // TODO: how to set depth here?
    
    double num_matched = 0;
    double num_unmatched = 0;
    
    for(int i = 0; i < merger->get_dat()->get_alphabet_size(); ++i){
        if(l->pos(i) != 0 && r->pos(i) != 0){
            num_matched++;
        } else {
            if(l->pos(i) != 0){
                num_unmatched++;
            }
        }
    }
    overlap += num_matched - num_unmatched;
};


double overlap_driven::compute_score(state_merger *merger, apta_node* left, apta_node* right){
  return (int) overlap;
};

void overlap_driven::reset(state_merger *merger){
  inconsistency_found = false;
  overlap = 0;
};



