#define STATS_GO_INLINE

#include <math.h>
#include <map>
#include "utility/stats.hpp"

#include "state_merger.h"
#include "evaluate.h"
#include "likelihood.h"
#include "parameters.h"

REGISTER_DEF_DATATYPE(likelihood_data);
REGISTER_DEF_TYPE(likelihoodratio);

likelihood_data::likelihood_data() : alergia_data() {
    undo_loglikelihood_orig = 0.0;
    undo_loglikelihood_merged = 0.0;
    undo_extra_parameters = 0;
};

void likelihood_data::initialize() {
    alergia_data::initialize();
    undo_loglikelihood_orig = 0.0;
    undo_loglikelihood_merged = 0.0;
    undo_extra_parameters = 0;
}

bool likelihoodratio::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    //likelihood_data* l = (likelihood_data*) left->get_data();
    //likelihood_data* r = (likelihood_data*) right->get_data();

    return count_driven::consistent(merger, left, right, depth);
};

void likelihoodratio::update_likelihood(double left_count, double right_count, double left_divider, double right_divider){
    //cerr << left_count << " " << right_count << " " << left_divider << " " << right_divider << endl;
    if (left_count == 0.0 && right_count == 0.0)
        return;

    if(right_count != 0.0 && left_count != 0.0)
        extra_parameters = extra_parameters + 1;

    if (left_count >= SYMBOL_COUNT && right_count >= SYMBOL_COUNT){
        left_count += CORRECTION;
        right_count += CORRECTION;

        if(left_count != 0.0)
            loglikelihood_orig += (left_count)  * log((left_count)  / left_divider);
        if(right_count != 0.0)
            loglikelihood_orig += (right_count) * log((right_count) / right_divider);
        if(right_count != 0.0 || left_count != 0.0)
            loglikelihood_merged += (left_count + right_count) * log((left_count + right_count) / (left_divider + right_divider));
    }
    //cerr << loglikelihood_orig << " " << loglikelihood_merged << endl;
};

void likelihoodratio::update_likelihood_pool(double left_count, double right_count, double left_divider, double right_divider){
    if (left_count == 0.0 && right_count == 0.0)
        return;

    if(right_count != 0.0 && left_count != 0.0)
        extra_parameters = extra_parameters + 1;

    if (left_count >= SYMBOL_COUNT || right_count >= SYMBOL_COUNT){
        left_count += CORRECTION;
        right_count += CORRECTION;

        if(left_count != 0.0)
            loglikelihood_orig += (left_count)  * log((left_count)  / left_divider);
        if(right_count != 0.0)
            loglikelihood_orig += (right_count) * log((right_count) / right_divider);
        if(right_count != 0.0 || left_count != 0.0)
            loglikelihood_merged += (left_count + right_count) * log((left_count + right_count) / (left_divider + right_divider));
        //if(right_count != 0.0 && left_count != 0.0)//if(left_count >= SYMBOL_COUNT && right_count >= SYMBOL_COUNT)
        //    extra_parameters = extra_parameters + 1;
    }
};

/* Likelihood Ratio (LR), computes an LR-test (used in RTI) and uses the p-value as score and consistency */
void likelihoodratio::update_score(state_merger *merger, apta_node* left, apta_node* right){
    likelihood_data* l = (likelihood_data*) left->get_data();
    likelihood_data* r = (likelihood_data*) right->get_data();

    double temp_loglikelihood_orig = loglikelihood_orig;
    double temp_loglikelihood_merged = loglikelihood_merged;
    int temp_extra_parameters = extra_parameters;

    /* we ignore low frequency states, decided by input parameter STATE_COUNT */
    if(FINAL_PROBABILITIES) {
        if (r->num_paths() + r->num_final() < STATE_COUNT || l->num_paths() + l->num_final() < STATE_COUNT) return;
    } else {
        if(r->num_paths() < STATE_COUNT || l->num_paths() < STATE_COUNT) return;
    }

        /* computing the dividers (denominator) */
        double left_divider = 0.0;
        double right_divider = 0.0;
        /* we pool low frequency counts (sum them up in a separate bin), decided by input parameter SYMBOL_COUNT
        * we create pools 1 and 2 separately for left and right low counts
        * in this way, we can detect differences in distributions even if all counts are low (i.e. [0,0,1,1] vs [1,1,0,0]) */
        double l1_pool = 0.0;
        double r1_pool = 0.0;
        double l2_pool = 0.0;
        double r2_pool = 0.0;

        int matching_right = 0;
        for (num_map::iterator it = l->symbol_counts.begin(); it != l->symbol_counts.end(); ++it) {
            int symbol = it->first;
            double left_count = it->second;
            if (left_count == 0) continue;

            double right_count = r->count(symbol);
            matching_right += right_count;

            update_divider(left_count, right_count, left_divider, right_divider);
            update_left_pool(left_count, right_count, l1_pool, r1_pool);
            update_right_pool(left_count, right_count, l2_pool, r2_pool);
        }
        r2_pool += r->num_paths() - matching_right;

        /* optionally add final probabilities (input parameter) */
        if (FINAL_PROBABILITIES) {
            double left_count = l->num_final();
            double right_count = r->num_final();

            update_divider(left_count, right_count, left_divider, right_divider);
            update_left_pool(left_count, right_count, l1_pool, r1_pool);
            update_right_pool(left_count, right_count, l2_pool, r2_pool);
        }

        update_divider_pool(l1_pool, r1_pool, left_divider, right_divider);
        update_divider_pool(l2_pool, r2_pool, left_divider, right_divider);

        if(left_divider < STATE_COUNT || right_divider < STATE_COUNT) return;

        /* now we have the dividers and pools, we compute the likelihoods */
        for (num_map::iterator it = l->symbol_counts.begin(); it != l->symbol_counts.end(); ++it) {
            int symbol = it->first;
            double left_count = it->second;
            double right_count = r->count(symbol);

            update_likelihood(left_count, right_count, left_divider, right_divider);
        }
        /* and final probabilities */
        if (FINAL_PROBABILITIES) update_likelihood(l->num_final(), r->num_final(), left_divider, right_divider);
        /* count the pools */
        update_likelihood_pool(l1_pool, r1_pool, left_divider, right_divider);
        update_likelihood_pool(l2_pool, r2_pool, left_divider, right_divider);

    r->undo_loglikelihood_orig = loglikelihood_orig - temp_loglikelihood_orig;
    r->undo_loglikelihood_merged = loglikelihood_merged - temp_loglikelihood_merged;
    r->undo_extra_parameters = extra_parameters - temp_extra_parameters;
};

void likelihoodratio::split_update_score_before(state_merger* merger, apta_node* left, apta_node* right, tail* t) {
    //likelihood_data *l = (likelihood_data *) left->get_data();
    likelihood_data *r = (likelihood_data *) right->get_data();

    loglikelihood_orig -= r->undo_loglikelihood_orig;
    loglikelihood_merged -= r->undo_loglikelihood_merged;
    extra_parameters -= r->undo_extra_parameters;

    r->undo_loglikelihood_orig = 0.0;
    r->undo_loglikelihood_merged = 0.0;
    r->undo_extra_parameters = 0;
};

void likelihoodratio::split_update_score_after(state_merger* merger, apta_node* left, apta_node* right, tail* t) {
    //likelihood_data *l = (likelihood_data *) left->get_data();
    //likelihood_data *r = (likelihood_data *) right->get_data();

    update_score(merger, left, right);
};

bool likelihoodratio::compute_consistency(state_merger *merger, apta_node* left, apta_node* right){
    double test_statistic = 2.0 * (loglikelihood_orig - loglikelihood_merged);
    double p_value = 1.0 - stats::pchisq(test_statistic, extra_parameters, false);

    if (p_value < CHECK_PARAMETER) { return false; }

    if (inconsistency_found) return false;

    return true;
};

double likelihoodratio::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    double test_statistic = 2.0 * (loglikelihood_orig - loglikelihood_merged);
    double p_value = 1.0 - stats::pchisq(test_statistic, extra_parameters, false);

    return p_value;
};

bool likelihoodratio::split_compute_consistency(state_merger *, apta_node* left, apta_node* right){
    double test_statistic = 2.0 * (loglikelihood_orig - loglikelihood_merged);
    double p_value = 1.0 - stats::pchisq(test_statistic, extra_parameters, false);

    if(left->get_size() <= STATE_COUNT || right->get_size() <= STATE_COUNT) return false;
    if(USE_SINKS && (left->get_size() <= SINK_COUNT || right->get_size() <= SINK_COUNT)) return false;
    if (p_value > CHECK_PARAMETER) return false;

    if (inconsistency_found) return false;

    return true;
};

double likelihoodratio::split_compute_score(state_merger *, apta_node* left, apta_node* right){
    double test_statistic = 2.0 * (loglikelihood_orig - loglikelihood_merged);
    double p_value = 1.0 - stats::pchisq(test_statistic, extra_parameters, false);

    return 1.0 + CHECK_PARAMETER - p_value;
};

void likelihoodratio::reset(state_merger *merger){
    inconsistency_found = false;
    loglikelihood_orig = 0;
    loglikelihood_merged = 0;
    extra_parameters = 0;
};


double likelihoodratio::compute_global_score(state_merger* m) {
    state_set* states = m->get_all_states();
    double likelihood = 0.0;
    double parameters = 0.0;
    for(state_set::iterator it = states->begin(); it != states->end(); ++it){
        likelihood_data* lldat = (likelihood_data*) (*it)->get_data();
            double divider = 0.0;
            for (num_map::iterator it = lldat->symbol_counts.begin(); it != lldat->symbol_counts.end(); ++it) {
                double count = it->second;
                divider += count;
            }
            if(FINAL_PROBABILITIES) divider += lldat->num_final();
            if(divider == 0) continue;

            for (num_map::iterator it = lldat->symbol_counts.begin(); it != lldat->symbol_counts.end(); ++it) {
                double count = it->second;
                if(count == 0) continue;

                likelihood += count * log(count / divider);
                parameters += 1.0;
            }
            if(FINAL_PROBABILITIES){
                double count = lldat->num_final();
                if(count != 0){
                    likelihood += count * log(count / divider);
                    parameters += 1.0;
                }
            }
        }
    return 2.0 * parameters - 2.0 * likelihood;
}

double likelihoodratio::compute_partial_score(state_merger* m) {
    state_set* states = m->get_red_states();
    double likelihood = 0.0;
    double parameters = 0.0;
    for(state_set::iterator it = states->begin(); it != states->end(); ++it){
        likelihood_data* lldat = (likelihood_data*) (*it)->get_data();
            double divider = 0.0;
            for (num_map::iterator it = lldat->symbol_counts.begin(); it != lldat->symbol_counts.end(); ++it) {
                double count = it->second;
                divider += count;
            }
            if(FINAL_PROBABILITIES) divider += lldat->num_final();
            if(divider == 0) continue;

            for (num_map::iterator it = lldat->symbol_counts.begin(); it != lldat->symbol_counts.end(); ++it) {
                double count = it->second;
                if(count == 0) continue;

                likelihood += count * log(count / divider);
                parameters += 1.0;
            }
            if(FINAL_PROBABILITIES){
                double count = lldat->num_final();
                if(count != 0){
                    likelihood += count * log(count / divider);
                    parameters += 1.0;
                }
            }
        }
    return 2.0 * parameters - 2.0 * likelihood;
}
