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

bool alergia94::pool_and_compute_tests(num_map& left_map, int left_total, int left_final,
                                     num_map& right_map, int right_total, int right_final) {
    /* computing the dividers (denominator) */
    double left_divider = (double) left_total;
    double right_divider = (double) right_total;

    if (FINAL_PROBABILITIES) {
        left_divider += (double) left_final;
        right_divider += (double) right_final;
    }

    for (auto & symbol_count : left_map) {
        int symbol = symbol_count.first;
        double left_count = symbol_count.second;
        if (left_count == 0) continue;

        double right_count = 0;
        auto hit = right_map.find(symbol);
        if(hit != right_map.end()) right_count = hit->second;

        if (!test_and_update(right_count, left_count, right_divider, left_divider)) {
            inconsistency_found = true;
            return false;
        }
    }

    /* count the final probabilities */
    if (FINAL_PROBABILITIES) {
        double left_count = left_final;
        double right_count = right_final;

        if (!test_and_update(right_count, left_count, right_divider, left_divider)) {
            inconsistency_found = true;
            return false;
        }
    }

    /* computing the remaining bins */
    for (auto & symbol_count : right_map) {
        int symbol = symbol_count.first;
        double right_count = symbol_count.second;
        if (right_count == 0) continue;

        double left_count = 0;
        auto hit = left_map.find(symbol);
        if(hit != left_map.end()) left_count = hit->second;
        if (left_count != 0) continue;

        if (!test_and_update(right_count, left_count, right_divider, left_divider)) {
            inconsistency_found = true;
            return false;
        }
    }
    return true;
};

double alergia94::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    return num_tests;
};
