#include <math.h>
#include <map>
#include "state_merger.h"
#include "evaluate.h"
#include "alergia94.h"
#include "parameters.h"

REGISTER_DEF_DATATYPE(alergia94_data);
REGISTER_DEF_TYPE(alergia94);

alergia94_data::alergia94_data(){
};

bool alergia94::alergia_consistency(double right_count, double left_count, double right_total, double left_total){
    double bound = (1.0 / sqrt(left_total) + 1.0 / sqrt(right_total));
    bound = bound * sqrt(0.5 * log(2.0 / CHECK_PARAMETER));
    
    double gamma = (left_count / left_total) - (right_count / right_total);
    
    if(gamma > bound) return false;
    if(-gamma > bound) return false;
    
    return true;
};

bool alergia94::data_consistent(alergia94_data* l, alergia94_data* r){
    /* we ignore low frequency states, decided by input parameter STATE_COUNT */
    if(FINAL_PROBABILITIES) {
        if (r->num_paths() + r->num_final() < STATE_COUNT ||
            l->num_paths() + l->num_final() < STATE_COUNT)
            return true;
    } else {
        if (r->num_paths() < STATE_COUNT || l->num_paths() < STATE_COUNT) return true;
    }

    /* computing the dividers (denominator) */
    double left_divider = (double) l->num_paths();
    double right_divider = (double) r->num_paths();

    if (FINAL_PROBABILITIES) {
        left_divider += (double) l->num_final();
        right_divider += (double) r->num_final();
    }

    for (auto & symbol_count : l->symbol_counts) {
        int symbol = symbol_count.first;
        double left_count = symbol_count.second;
        if (left_count == 0) continue;
        double right_count = r->count(symbol);

        if (!alergia_consistency(right_count, left_count, right_divider, left_divider)) {
            inconsistency_found = true;
            return false;
        }
    }

    /* count the final probabilities */
    if (FINAL_PROBABILITIES) {
        double left_count = l->num_final();
        double right_count = r->num_final();

        if (!alergia_consistency(right_count, left_count, right_divider, left_divider)) {
            inconsistency_found = true;
            return false;
        }
    }

    /* computing the remaining bins */
    for (auto & symbol_count : r->symbol_counts) {
        int symbol = symbol_count.first;
        double right_count = symbol_count.second;
        if (right_count == 0) continue;
        double left_count = l->count(symbol);
        if (left_count != 0) continue;

        if (!alergia_consistency(right_count, left_count, right_divider, left_divider)) {
            inconsistency_found = true;
            return false;
        }
    }
    return true;
};

/* ALERGIA, consistency based on Hoeffding bound */
bool alergia94::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    if(!count_driven::consistent(merger, left, right, depth)){ inconsistency_found = true; return false; }
    auto* l = dynamic_cast<alergia94_data*>(left->get_data());
    auto* r = dynamic_cast<alergia94_data*>(right->get_data());
    
    return data_consistent(l, r);
};
