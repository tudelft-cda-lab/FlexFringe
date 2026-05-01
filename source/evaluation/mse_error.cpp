#include "state_merger.h"
#include "mse_error.h"
#include <math.h>
#include <map>

#include "catch.hpp"
#include "parameters.h"
#include "input/inputdatalocator.h"

REGISTER_DEF_DATATYPE(mse_data);
REGISTER_DEF_TYPE(mse_error);

void mse_data::print_state_label(std::iostream& output){
    evaluation_data::print_state_label(output);
    int attr_index = 0;
    for(int attr = 0; attr < inputdata_locator::get()->get_num_attributes(); ++attr) {
        if(!inputdata_locator::get()->is_target(attr)) continue;
        output << "\n" << inputdata_locator::get()->get_attribute(attr)->get_name() << " : " << (sums[attr_index] / num_tails) << " " << (sum_squares[attr_index] / num_tails);
        output << "\n" << " error : " << sum_squares[attr] - num_tails*((sums[attr_index] / num_tails)*(sums[attr_index] / num_tails));

        attr_index++;
    }
};


mse_data::mse_data(){
    num_tails = 0.0;
    for(int attr = 0; attr < inputdata_locator::get()->get_num_attributes(); ++attr) {
        if(!inputdata_locator::get()->is_target(attr)) continue;
        sums.push_back(0.0);
        sum_squares.push_back(0.0);
    }
    undo_rss_before = 0.0;
    undo_rss_after = 0.0;
    undo_num_points = 0;
    undo_total_merges = 0.0;
};

void mse_data::initialize() {
    num_tails = 0.0;
    sums.clear();
    sum_squares.clear();
    for(int attr = 0; attr < inputdata_locator::get()->get_num_attributes(); ++attr) {
        if(!inputdata_locator::get()->is_target(attr)) continue;
        sums.push_back(0.0);
        sum_squares.push_back(0.0);
    }
    undo_rss_before = 0.0;
    undo_rss_after = 0.0;
    undo_num_points = 0;
    undo_total_merges = 0.0;
}

void mse_data::add_tail(tail *t) {
    int attr_index = 0;
    for(int attr = 0; attr < inputdata_locator::get()->get_num_attributes(); ++attr) {
        if(!inputdata_locator::get()->is_target(attr)) continue;
        double occ = t->get_value(attr);
        num_tails += 1.0;
        sums[attr_index] = sums[attr_index] + occ;
        sum_squares[attr_index] = sum_squares[attr_index] + occ*occ;
        attr_index++;
    }
};

void mse_data::del_tail(tail *t) {
    int attr_index = 0;
    for(int attr = 0; attr < inputdata_locator::get()->get_num_attributes(); ++attr) {
        if(!inputdata_locator::get()->is_target(attr)) continue;
        double occ = t->get_value(attr);
        num_tails -= 1.0;
        sums[attr_index] = sums[attr_index] - occ;
        sum_squares[attr_index] = sum_squares[attr_index] - occ*occ;
        attr_index++;
    }
};

void mse_data::update(evaluation_data* right){
    mse_data* r = (mse_data*) right;
    num_tails += r->num_tails;
    for (int i = 0; i < sums.size(); ++i) {
        sums[i] += r->sums[i];
        sum_squares[i] += r->sum_squares[i];
    }
    // if(r->occs.size() != 0)
    //     mean = ((mean * ((double)occs.size()) + (r->mean * ((double)r->occs.size())))) / ((double)occs.size() + (double)r->occs.size());
    //
    // if(occs.size() != 0){
    //     r->merge_point = occs.end();
    //     --(r->merge_point);
    //     occs.splice(occs.end(), r->occs);
    //     ++(r->merge_point);
    // } else {
    //     occs.splice(occs.begin(), r->occs);
    //     r->merge_point = occs.begin();
    // }
};

void mse_data::undo(evaluation_data* right){
    mse_data* r = (mse_data*) right;
    num_tails -= r->num_tails;
    for (int i = 0; i < sums.size(); ++i) {
        sums[i] -= r->sums[i];
        sum_squares[i] -= r->sum_squares[i];
    }
    // r->occs.splice(r->occs.begin(), occs, r->merge_point, occs.end());
    //
    // if(occs.size() != 0)// && r->occs.size() != 0)
    //     mean = ((mean * ((double)occs.size() + (double)r->occs.size())) - (r->mean * ((double)r->occs.size()))) / ((double)occs.size());
    // else
    //     mean = 0;
};

bool mse_error::consistent(state_merger *merger, apta_node* left, apta_node* right){
    if(evaluation_function::consistent(merger, left, right) == false){ inconsistency_found = true; return false; }
    const auto* l = static_cast<mse_data *>(left->get_data());
    const auto* r = static_cast<mse_data *>(right->get_data());

    if(l->num_tails < SYMBOL_COUNT || r->num_tails < SYMBOL_COUNT) return true;
    for (int i = 0; i < l->sums.size(); ++i) {
        double mean_left = l->sums[i] / l->num_tails;
        double variance_left = l->sum_squares[i]/l->num_tails - (mean_left * mean_left);
        double mean_right = r->sums[i] / r->num_tails;
        double variance_right = r->sum_squares[i]/r->num_tails - (mean_right * mean_right);
        if(variance_left > 2.0 * variance_right + 0.01){ inconsistency_found = true; return false; }
        if(variance_right > 2.0 * variance_left + 0.01){ inconsistency_found = true; return false; }
    }
    return true;
};

void mse_error::update_score(state_merger *merger, apta_node* left, apta_node* right){
    mse_data* l = (mse_data*) left->get_data();
    mse_data* r = (mse_data*) right->get_data();
    
    double temp_RSS_before = RSS_before;
    double temp_RSS_after = RSS_after;
    double temp_num_points = num_points;

    if(l->num_tails <= STATE_COUNT || r->num_tails <= STATE_COUNT) return;

    bool already_merged = false;

    total_merges = total_merges + 1;

    if(already_merged)
        num_points += r->num_tails;
    else
        num_points += l->num_tails + r->num_tails;

    for (int i = 0; i < l->sums.size(); ++i) {
        double mean_left = l->sums[i] / l->num_tails;
        double mean_right = r->sums[i] / r->num_tails;
        double mean_total = (l->sums[i] + r->sums[i]) / (l->num_tails + r->num_tails);

        double error_left = l->sum_squares[i] - l->num_tails*(mean_left * mean_left);
        double error_right = r->sum_squares[i] - r->num_tails*(mean_right * mean_right);
        double error_total = (l->sum_squares[i] + r->sum_squares[i]) - (l->num_tails + r->num_tails)*(mean_total * mean_total);

        std::cerr << "error_left " << error_left << std::endl;
        std::cerr << "error_right " << error_right << std::endl;
        std::cerr << "error_total " << error_total << std::endl;

        if(already_merged){
            RSS_before += error_right;
            RSS_after  += error_total - error_left;
        } else {
            RSS_before += error_right+error_left;
            RSS_after  += error_total;
        }
    }

    r->undo_rss_before = RSS_before - temp_RSS_before;
    r->undo_rss_after = RSS_after - temp_RSS_after;
    r->undo_num_points = num_points - temp_num_points;
    r->undo_total_merges = 1;

    // double error_left = 0.0;
    // double error_right = 0.0;
    // double error_total = 0.0;
    //
    // for(double_list::iterator it = l->occs.begin(); it != l->occs.end(); ++it){
    //     error_left  = error_left  + ((l->mean    - (double)*it)*(l->mean    - (double)*it));
    //     error_total = error_total + ((mean_total - (double)*it)*(mean_total - (double)*it));
    // }
    // for(double_list::iterator it = r->occs.begin(); it != r->occs.end(); ++it){
    //     error_right = error_right + ((r->mean    - (double)*it)*(r->mean    - (double)*it));
    //     error_total = error_total + ((mean_total - (double)*it)*(mean_total - (double)*it));
    // }
};

// double compute_RSS(apta_node* node){
//     mse_data* l = (mse_data*) node->get_data();
//     double error = 0.0;
//
//     for(double_list::iterator it = l->occs.begin(); it != l->occs.end(); ++it){
//         error  += ((l->mean    - (double)*it)*(l->mean    - (double)*it));
//     }
//
//     return error;
// };

double mse_error::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    std::cerr << "computed merge " << num_points << " " << RSS_before << " " << RSS_after << std::endl;
    if (num_points == 0){ return -1.0; }
    if (RSS_before == 0 && RSS_after != 0){ return -1.0; }
    if (RSS_after == 0){ return 0.0; }
    return 2*total_merges + num_points*(log(RSS_before/num_points)) - num_points*log(RSS_after/num_points);
};

bool mse_error::compute_consistency(state_merger *merger, apta_node* left, apta_node* right){
    if (num_points == 0){ return false; }
    if (RSS_before == 0 && RSS_after != 0){ return false; }
    if (RSS_after == 0){ return true; }
    return 2*total_merges + num_points*(log(RSS_before/num_points)) - num_points*log(RSS_after/num_points) > CHECK_PARAMETER;
};

void mse_error::reset(state_merger *merger ){
    inconsistency_found = false;
    num_merges = 0.0;
    num_points = 0.0;
    RSS_before = 0.0;
    RSS_after = 0.0;
    total_merges = 0;
    prev_AIC = 0.0;

    aic_states.clear();
};

void mse_error::split_update_score_before(state_merger* merger, apta_node* left, apta_node* right, tail* t) {
    mse_data* l = (mse_data*) left->get_data();
    mse_data* r = (mse_data*) right->get_data();

    RSS_before -= r->undo_rss_before;
    RSS_after -= r->undo_rss_after;
    num_points -= r->undo_num_points;
    total_merges -= r->undo_total_merges;

    r->undo_rss_before = 0.0;
    r->undo_rss_after = 0.0;
    r->undo_num_points = 0;
    r->undo_total_merges = 0;
};

void mse_error::split_update_score_after(state_merger* merger, apta_node* left, apta_node* right, tail* t) {
    update_score(merger, left, right);
};

double mse_error::split_compute_score(state_merger *, apta_node* left, apta_node* right){
    std::cerr << "computed split " << num_points << " " << RSS_before << " " << RSS_after << std::endl;
    if (num_points == 0){ return -1.0; }
    if (RSS_before == 0 && RSS_after != 0){ return 0.0; }
    if (RSS_after == 0){ return -1.0; }
    return -2*total_merges - num_points*(log(RSS_before/num_points)) + num_points*log(RSS_after/num_points) + 100;
};

bool mse_error::split_compute_consistency(state_merger *, apta_node* left, apta_node* right){
    if (num_points == 0){ return false; }
    if (RSS_before == 0 && RSS_after != 0){ return true; }
    if (RSS_after == 0){ return false; }
    return -2*total_merges - num_points*(log(RSS_before/num_points)) + num_points*log(RSS_after/num_points) > -100;
};

bool is_low_occ_sink(apta_node* node){
    mse_data* l = (mse_data*) node->get_data();
    return l->num_tails < STATE_COUNT;
}

int mse_error::sink_type(apta_node* node){
    if(!USE_SINKS) return -1;

    if (is_low_occ_sink(node)) return 0;
    return -1;
};

bool mse_error::sink_consistent(apta_node* node, int type){
    if(!USE_SINKS) return true;
    
    if(type == 0) return is_low_occ_sink(node);
    return true;
};

int mse_error::num_sink_types(){
    if(!USE_SINKS) return 0;
    return 1;
};