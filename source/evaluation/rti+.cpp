
#define STATS_GO_INLINE

#include <math.h>
#include <vector>
#include <map>

#include "utility/stats.hpp"

#include "state_merger.h"
#include "evaluate.h"
#include "rti+.h"
#include "parameters.h"

REGISTER_DEF_DATATYPE(rtiplus_data);
REGISTER_DEF_TYPE(rtiplus);

vector< vector<double> > rtiplus::attribute_quantiles;

rtiplus_data::rtiplus_data() : likelihood_data::likelihood_data() {
    for(int i = 0; i < inputdata::get_num_attributes(); ++i){
        if(inputdata::is_distributionable(i)){
            if(QUANTILE_DISTRIBUTIONS) statistics.push_back(vector<double>(4, 0));
            else if(NORMAL_DISTRIBUTIONS) statistics.push_back(vector<double>(3, 0));
        }
    }
};

void rtiplus_data::initialize() {
    likelihood_data::initialize();
    int modifier = 0;
    for(int i = 0; i < inputdata::get_num_attributes(); ++i){
        if(!inputdata::is_distributionable(i)){
            ++modifier;
            continue;
        }
        if(QUANTILE_DISTRIBUTIONS) statistics[i - modifier].assign(4, 0);
        else if(NORMAL_DISTRIBUTIONS) statistics[i - modifier].assign(3, 0);
    }
};

void rtiplus_data::add_tail(tail* t){
    alergia_data::add_tail(t);
    int modifier = 0;

    for(int i = 0; i < inputdata::get_num_attributes(); ++i) {
        if(!inputdata::is_distributionable(i)){
            ++modifier;
            continue;
        }
        int attr = i-modifier;

        if(QUANTILE_DISTRIBUTIONS){
            bool found = false;
            for(int j = 0; j < rtiplus::attribute_quantiles[attr].size(); ++j){
                if(t->get_value(i) < rtiplus::attribute_quantiles[attr][j]){
                    statistics[attr][j] = statistics[attr][j] + 1;
                    found = true;
                    break;
                }
            }
            if(!found){
                statistics[attr][rtiplus::attribute_quantiles[attr].size()] = statistics[attr][rtiplus::attribute_quantiles[attr].size()] + 1;
            }
        }
        else if(NORMAL_DISTRIBUTIONS){
            // NUMBER OF ITEMS
            statistics[attr][0] = statistics[attr][0] + 1;
            // SUM OF VALUES
            statistics[attr][1] = statistics[attr][1] + t->get_value(i);
            // SQUARED SUMS
            statistics[attr][2] = statistics[attr][2] + (t->get_value(i) * t->get_value(i));
        }
    }
};

void rtiplus_data::del_tail(tail* t){
    alergia_data::del_tail(t);
    int modifier = 0;

    for(int i = 0; i < inputdata::get_num_attributes(); ++i) {
        if(!inputdata::is_distributionable(i)){
            ++modifier;
            continue;
        }
        int attr = i-modifier;

        if(QUANTILE_DISTRIBUTIONS){
            bool found = false;
            for(int j = 0; j < rtiplus::attribute_quantiles[attr].size(); ++j){
                if(t->get_value(i) < rtiplus::attribute_quantiles[attr][j]){
                    statistics[attr][j] = statistics[attr][j] - 1;
                    found = true;
                    break;
                }
            }
            if(!found){
                statistics[attr][rtiplus::attribute_quantiles[attr].size()] = statistics[attr][rtiplus::attribute_quantiles[attr].size()] - 1;
            }
        }
        else if(NORMAL_DISTRIBUTIONS){
            // NUMBER OF ITEMS
            statistics[attr][0] = statistics[attr][0] - 1;
            // SUM OF VALUES
            statistics[attr][1] = statistics[attr][1] - t->get_value(i);
            // SQUARED SUMS
            statistics[attr][2] = statistics[attr][2] - (t->get_value(i) * t->get_value(i));
        }
    }
};

void rtiplus_data::print_state_label(iostream& output){
    likelihood_data::print_state_label(output);
    output << endl;
    if(QUANTILE_DISTRIBUTIONS){
        for(int i = 0; i < statistics.size(); ++i) {
            output << inputdata::get_attribute(i)->get_name() << ":[";
            for(int j = 0; j < rtiplus::attribute_quantiles[i].size()+1; ++j){
                output << statistics[i][j] << ",";
            }
            output << "]" << endl;
        }
    } else if(NORMAL_DISTRIBUTIONS){
        int modifier = 0;
        for(int i = 0; i < inputdata::get_num_attributes(); ++i) {
            if (!inputdata::is_distributionable(i)) {
                ++modifier;
                continue;
            }
            int attr = i - modifier;
            output << inputdata::get_attribute(i)->get_name() << ":";
            double mean = statistics[attr][1] / statistics[attr][0];
            double var = statistics[attr][2] / statistics[attr][0] - (mean * mean);
            output << " m(" << mean << ") sd(" << sqrt(var) << ")" << endl;
        }
    }
};

double rtiplus_data::predict_score(tail* t){
    double result = alergia_data::predict_symbol_score(t->get_symbol());
    int modifier = 0;
    double divider = (double) num_paths();
    if(divider == 0) return result;

    for(int i = 0; i < inputdata::get_num_attributes(); ++i) {
        if (!inputdata::is_distributionable(i)) {
            ++modifier;
            continue;
        }

        int attr = i - modifier;
        double val = t->get_value(i);
        if (QUANTILE_DISTRIBUTIONS) {
            bool found = false;
            for(int j = 0; j < rtiplus::attribute_quantiles[attr].size(); ++j){
                if(val < rtiplus::attribute_quantiles[attr][j]){
                    result += log(((double) statistics[attr][j]) / divider);
                    found = true;
                    break;
                }
            }
            if(!found){
                result += log(((double) statistics[attr][rtiplus::attribute_quantiles[attr].size()]) / divider);
            }
        } else if (NORMAL_DISTRIBUTIONS) {
            double mean = statistics[attr][1] / statistics[attr][0];
            double var = statistics[attr][2] / statistics[attr][0] - (mean * mean);
            if(var < 0.1) var = 0.1;
            double prob = stats::dnorm(val,mean,sqrt(var), false);
            cerr << val << " " << mean << " " << sqrt(var) << " " << prob << endl;
            result += log(prob);
        }
    }
    return result;
};

void rtiplus_data::read_json(json& data){
    alergia_data::read_json(data);

    json& d = data["rti_statistics"];
    for(int i = 0; i < statistics.size(); ++i) {
        for(int j = 0; j < statistics[0].size(); ++j) {
            statistics[i][j] = d[i][j];
        }
    }
};

void rtiplus_data::write_json(json& data){
    alergia_data::write_json(data);
    data["rti_statistics"] = statistics;
};

void rtiplus_data::update(evaluation_data* right){
    likelihood_data::update(right);
    rtiplus_data* other = (rtiplus_data*)right;
    for(int i = 0; i < statistics.size(); ++i) {
        for(int j = 0; j < rtiplus::attribute_quantiles[i].size()+1; ++j){
            statistics[i][j] += other->statistics[i][j];
        }
    }
};

void rtiplus_data::undo(evaluation_data* right){
    likelihood_data::undo(right);
    rtiplus_data* other = (rtiplus_data*)right;
    for(int i = 0; i < statistics.size(); ++i) {
        for(int j = 0; j < rtiplus::attribute_quantiles[i].size()+1; ++j){
            statistics[i][j] -= other->statistics[i][j];
        }
    }
};

void rtiplus_data::split_update(evaluation_data* right){
    undo(right);
};

void rtiplus_data::split_undo(evaluation_data* right){
    update(right);
};

/* Likelihood Ratio (LR), computes an LR-test (used in RTI) and uses the p-value as score and consistency */
void rtiplus::update_score(state_merger *merger, apta_node* left, apta_node* right){
    double temp_loglikelihood_orig = loglikelihood_orig;
    double temp_loglikelihood_merged = loglikelihood_merged;
    int temp_extra_parameters = extra_parameters;

    likelihoodratio::update_score(merger, left, right);

    rtiplus_data* l = (rtiplus_data*) left->get_data();
    rtiplus_data* r = (rtiplus_data*) right->get_data();


    /* we ignore low frequency states, decided by input parameter STATE_COUNT */
    if(FINAL_PROBABILITIES){
        if(r->num_paths() + r->num_final() < STATE_COUNT || l->num_paths() + l->num_final() < STATE_COUNT) return;
    } else {
        if(r->num_paths() < STATE_COUNT || l->num_paths() < STATE_COUNT) return;
    }

    if(QUANTILE_DISTRIBUTIONS){
        for(int i = 0; i < l->statistics.size(); ++i) {
            /* computing the dividers (denominator) */
            double left_divider = 0.0;
            double right_divider = 0.0;
            double left_count = 0.0;
            double right_count  = 0.0;

            double l1_pool = 0.0;
            double r1_pool = 0.0;
            double l2_pool = 0.0;
            double r2_pool = 0.0;

            for (int j = 0; j < rtiplus::attribute_quantiles[i].size() + 1; ++j) {
                left_count = l->statistics[i][j];
                right_count = r->statistics[i][j];

                update_divider(left_count, right_count, left_divider, right_divider);
                update_left_pool(left_count, right_count, l1_pool, r1_pool);
                update_right_pool(left_count, right_count, l2_pool, r2_pool);
            }

            update_divider_pool(l1_pool, r1_pool, left_divider, right_divider);
            update_divider_pool(l2_pool, r2_pool, left_divider, right_divider);

            if(left_divider < STATE_COUNT || right_divider < STATE_COUNT) continue;

            for (int j = 0; j < rtiplus::attribute_quantiles[i].size() + 1; ++j) {
                left_count = l->statistics[i][j];
                right_count = r->statistics[i][j];

                likelihoodratio::test_and_update(left_count, right_count, left_divider, right_divider);
            }
            update_likelihood_pool(l1_pool, r1_pool, left_divider, right_divider);
            update_likelihood_pool(l2_pool, r2_pool, left_divider, right_divider);
        }
    }
    else if(NORMAL_DISTRIBUTIONS){
        int modifier = 0;
        for(int i = 0; i < inputdata::get_num_attributes(); ++i) {
            if(!inputdata::is_distributionable(i)){
                ++modifier;
                continue;
            }
            int attr = i-modifier;

            if(l->statistics[attr][0] == 0) continue;
            if(r->statistics[attr][0] == 0) continue;

            double mean_left = l->statistics[attr][1] / l->statistics[attr][0];
            double var_left  = l->statistics[attr][2] / l->statistics[attr][0] - (mean_left * mean_left);

            double mean_right = r->statistics[attr][1] / r->statistics[attr][0];
            double var_right  = r->statistics[attr][2] / r->statistics[attr][0] - (mean_right * mean_right);

            double mean_total = (l->statistics[attr][1] + r->statistics[attr][1]) / (l->statistics[attr][0] + r->statistics[attr][0]);
            double var_total  = (l->statistics[attr][2] + r->statistics[attr][2]) / (l->statistics[attr][0] + r->statistics[attr][0]) - (mean_total * mean_total);

            if(var_right < 0.1) var_right = 0.1;
            if(var_left < 0.1) var_left = 0.1;
            if(var_total < 0.1) var_total = 0.1;

            for(auto it = tail_iterator(left); *it != nullptr; ++it) {
                //cerr << (*it)->get_value(i) << endl;
                //cerr << log(stats::dnorm((*it)->get_value(i), mean_left, sqrt(var_left))) << endl;
                //cerr << log(stats::dnorm((*it)->get_value(i), mean_total, sqrt(var_total))) << endl;
                loglikelihood_orig    += log(stats::dnorm((*it)->get_value(i), mean_left, sqrt(var_left)));
                loglikelihood_merged  += log(stats::dnorm((*it)->get_value(i), mean_total, sqrt(var_total)));
                cerr << loglikelihood_orig << " " << loglikelihood_merged << endl;
            }
            for(auto it = tail_iterator(right); *it != nullptr; ++it) {
                //cerr << (*it)->get_value(i) << endl;
                //cerr << log(stats::dnorm((*it)->get_value(i), mean_right, sqrt(var_right))) << endl;
                //cerr << log(stats::dnorm((*it)->get_value(i), mean_total, sqrt(var_total))) << endl;
                loglikelihood_orig    += log(stats::dnorm((*it)->get_value(i), mean_right, sqrt(var_right)));
                loglikelihood_merged  += log(stats::dnorm((*it)->get_value(i), mean_total, sqrt(var_total)));
                cerr << loglikelihood_orig << " " << loglikelihood_merged << endl;
            }
            cerr << "*****" << endl;
            /*
            loglikelihood_orig   -= l->statistics[i][0] * (log(2.0 * M_PI*var_left)) / 2.0;
            loglikelihood_orig   -= r->statistics[i][0] * (log(2.0 * M_PI*var_right)) / 2.0;
            loglikelihood_merged -= (l->statistics[i][0] + r->statistics[i][0]) * (log(M_PI*var_total)) / 2.0;
            for(auto it = tail_iterator(left); *it != nullptr; ++it){
                double diff = (*it)->get_value(i) - mean_left;
                loglikelihood_orig  -= (diff * diff) / (2.0 * var_left);
                diff = (*it)->get_value(i) - mean_total;
                loglikelihood_merged -= (diff * diff) / (2.0 * var_total);
            }
            for(auto it = tail_iterator(right); *it != nullptr; ++it){
                double diff = (*it)->get_value(i) - mean_right;
                loglikelihood_orig  -= (diff * diff) / (2.0 * var_right);
                diff = (*it)->get_value(i) - mean_total;
                loglikelihood_merged -= (diff * diff) / (2.0 * var_total);
            }
             */
            extra_parameters += 2;
        }
    }

    r->undo_loglikelihood_orig = loglikelihood_orig - temp_loglikelihood_orig;
    r->undo_loglikelihood_merged = loglikelihood_merged - temp_loglikelihood_merged;
    r->undo_extra_parameters = extra_parameters - temp_extra_parameters;
};

void rtiplus::split_update_score_before(state_merger* merger, apta_node* left, apta_node* right, tail* t) {
    //rtiplus_data *l = (rtiplus_data *) left->get_data();
    rtiplus_data *r = (rtiplus_data *) right->get_data();

    loglikelihood_orig -= r->undo_loglikelihood_orig;
    loglikelihood_merged -= r->undo_loglikelihood_merged;
    extra_parameters -= r->undo_extra_parameters;

    r->undo_loglikelihood_orig = 0.0;
    r->undo_loglikelihood_merged = 0.0;
    r->undo_extra_parameters = 0;
};

void rtiplus::split_update_score_after(state_merger* merger, apta_node* left, apta_node* right, tail* t) {
    update_score(merger, left, right);
};

bool rtiplus::split_compute_consistency(state_merger *, apta_node* left, apta_node* right){
    double test_statistic = 2.0 * (loglikelihood_orig - loglikelihood_merged);
    double p_value = 1.0 - stats::pchisq(test_statistic, extra_parameters, false);

    if(left->get_size() <= STATE_COUNT || right->get_size() <= STATE_COUNT) return false;
    if(USE_SINKS && (left->get_size() <= SINK_COUNT || right->get_size() <= SINK_COUNT)) return false;
    if (p_value > CHECK_PARAMETER) return false;
    
    if (inconsistency_found) return false;
    
    return true;
};

double rtiplus::split_compute_score(state_merger *, apta_node* left, apta_node* right){
    double test_statistic = 2.0 * (loglikelihood_orig - loglikelihood_merged);
    double p_value = 1.0 - stats::pchisq(test_statistic, extra_parameters, false);

    return 1.0 + CHECK_PARAMETER - p_value;
};

void rtiplus::initialize_after_adding_traces(state_merger* merger){
};

void rtiplus::initialize_before_adding_traces(){
    int corr = 0;
    for(int a = 0; a < merger->get_dat()->get_num_attributes(); ++a){
        if(!merger->get_dat()->is_distributionable(a)){
            corr += 1;
            continue;
        }

        rtiplus::attribute_quantiles.emplace_back(3,0.0);
        multiset<double> values;
        for(auto it = merger->get_dat()->traces_start();
            it != merger->get_dat()->traces_end(); ++it){
            for(tail* t = (*it)->get_head(); t != (*it)->get_end(); t = t->future()){
                values.insert(t->get_value(a));
                //cerr << " " << t->get_value(a);
            }
        }

        int Q1 = (int)round((double)values.size() / 4.0);
        int Q2 = Q1 + Q1;
        int Q3 = Q2 + Q1;

        int V1 = 0;
        int V2 = 0;
        int V3 = 0;

        int count = 0;
        for(double value : values){
            if(count == Q1) V1 = value;
            if(count == Q2) V2 = value;
            if(count == Q3) V3 = value;
            count = count + 1;
        }

        rtiplus::attribute_quantiles[a-corr][0] = V1;
        rtiplus::attribute_quantiles[a-corr][1] = V2;
        rtiplus::attribute_quantiles[a-corr][2] = V3;
    }
}

void rtiplus::reset_split(state_merger *merger, apta_node* node){
    inconsistency_found = false;
    loglikelihood_orig = 0;
    loglikelihood_merged = 0;
    extra_parameters = 0;
};

void rtiplus::read_json(json& data){
    alergia::read_json(data);

    if(QUANTILE_DISTRIBUTIONS){
        json& d = data["rti_quantiles"];
        for(int i = 0; i < d.size(); ++i) {
            attribute_quantiles.push_back(d[i]);
        }
    }
};

void rtiplus::write_json(json& data){
    alergia::write_json(data);
    if(QUANTILE_DISTRIBUTIONS){
        data["rti_quantiles"] = attribute_quantiles;
    }
};

