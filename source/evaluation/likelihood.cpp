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
    undo_total_count = 0;
};

void likelihood_data::initialize() {
    alergia_data::initialize();
    undo_loglikelihood_orig = 0.0;
    undo_loglikelihood_merged = 0.0;
    undo_extra_parameters = 0;
    undo_total_count = 0;
}

void likelihood_data::print_state_label(iostream& output){
    alergia_data::print_state_label(output);
};

bool likelihoodratio::consistent(state_merger *merger, apta_node* left, apta_node* right){
    return count_driven::consistent(merger, left, right);
};

bool likelihoodratio::test_and_update(double right_count, double left_count, double right_divider, double left_divider) {
    //cerr << "update likelihood - " << left_count << " " << right_count << " " << left_divider << " " << right_divider << endl;
    if (left_count == 0.0 && right_count == 0.0) return true;

    if(right_count != 0.0 && left_count != 0.0)
        extra_parameters = extra_parameters + 1;
    total_count += left_count + right_count;

    if (left_count >= SYMBOL_COUNT || right_count >= SYMBOL_COUNT){
        left_count += CORRECTION;
        right_count += CORRECTION;

        if(left_count != 0.0)
            loglikelihood_orig += (left_count)  * log((left_count)  / left_divider);
        if(right_count != 0.0)
            loglikelihood_orig += (right_count) * log((right_count) / right_divider);
        if(right_count != 0.0 || left_count != 0.0)
            loglikelihood_merged += (left_count + right_count) * log((left_count + right_count) / (left_divider + right_divider));
    }
    //cerr << "likelihoods - " << loglikelihood_orig << " " << loglikelihood_merged << endl;
    return true;
};

void likelihoodratio::update_likelihood_pool(double left_count, double right_count, double left_divider, double right_divider){
    if (left_count == 0.0 && right_count == 0.0)
        return;

    if(right_count != 0.0 && left_count != 0.0)
        extra_parameters = extra_parameters + 1;

    total_count += left_count + right_count;

    if (left_count >= SYMBOL_COUNT || right_count >= SYMBOL_COUNT){
        left_count += CORRECTION;
        right_count += CORRECTION;

        if(left_count != 0.0)
            loglikelihood_orig += (left_count)  * log((left_count)  / left_divider);
        if(right_count != 0.0)
            loglikelihood_orig += (right_count) * log((right_count) / right_divider);
        if(right_count != 0.0 || left_count != 0.0)
            loglikelihood_merged += (left_count + right_count) * log((left_count + right_count) / (left_divider + right_divider));
    }
};

/* Likelihood Ratio (LR), computes an LR-test (used in RTI) and uses the p-value as score and consistency */
void likelihoodratio::update_score(state_merger *merger, apta_node* left, apta_node* right) {
    likelihood_data *l = (likelihood_data *) left->get_data();
    likelihood_data *r = (likelihood_data *) right->get_data();

    double temp_loglikelihood_orig = loglikelihood_orig;
    double temp_loglikelihood_merged = loglikelihood_merged;
    int temp_extra_parameters = extra_parameters;
    int temp_total_count = total_count;

    prob_consistency(l, r);

    r->undo_loglikelihood_orig = loglikelihood_orig - temp_loglikelihood_orig;
    r->undo_loglikelihood_merged = loglikelihood_merged - temp_loglikelihood_merged;
    r->undo_extra_parameters = extra_parameters - temp_extra_parameters;
    r->undo_total_count = total_count - temp_total_count;
}

void likelihoodratio::split_update_score_before(state_merger* merger, apta_node* left, apta_node* right, tail* t) {
    likelihood_data *r = (likelihood_data *) right->get_data();

    loglikelihood_orig -= r->undo_loglikelihood_orig;
    loglikelihood_merged -= r->undo_loglikelihood_merged;
    extra_parameters -= r->undo_extra_parameters;
    total_count -= r->undo_total_count;

    r->undo_loglikelihood_orig = 0.0;
    r->undo_loglikelihood_merged = 0.0;
    r->undo_extra_parameters = 0;
    r->undo_total_count  = 0;
};

void likelihoodratio::split_update_score_after(state_merger* merger, apta_node* left, apta_node* right, tail* t) {
    update_score(merger, left, right);
};

bool likelihoodratio::compute_consistency(state_merger *merger, apta_node* left, apta_node* right){
    //cerr << "test: " << 2.0 * (loglikelihood_orig - loglikelihood_merged) << endl;

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
    total_count = 0;
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
