#include "state_merger.h"
#include "mse_error.h"
#include <math.h>
#include <map>
#include "parameters.h"

REGISTER_DEF_DATATYPE(mse_data);
REGISTER_DEF_TYPE(mse_error);

mse_data::mse_data(){
    mean = 0;
};

void mse_data::add_tail(tail *t) {
    double occ = std::stod(t->get_data());
    mean = ((mean * ((double)occs.size())) + occ) / ((double)(occs.size() + 1));
    occs.push_front(occ);
};

void mse_data::update(evaluation_data* right){
    mse_data* r = (mse_data*) right;
    
    if(r->occs.size() != 0)
        mean = ((mean * ((double)occs.size()) + (r->mean * ((double)r->occs.size())))) / ((double)occs.size() + (double)r->occs.size());
    
    if(occs.size() != 0){
        r->merge_point = occs.end();
        --(r->merge_point);
        occs.splice(occs.end(), r->occs);
        ++(r->merge_point);
    } else {
        occs.splice(occs.begin(), r->occs);
        r->merge_point = occs.begin();
    }
};

void mse_data::undo(evaluation_data* right){
    mse_data* r = (mse_data*) right;

    r->occs.splice(r->occs.begin(), occs, r->merge_point, occs.end());
    
    if(occs.size() != 0)// && r->occs.size() != 0)
        mean = ((mean * ((double)occs.size() + (double)r->occs.size())) - (r->mean * ((double)r->occs.size()))) / ((double)occs.size());
    else
        mean = 0;
};

string mse_data::predict_data(tail*){
    return to_string(mean);
};

bool mse_error::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    if(evaluation_function::consistent(merger, left, right, depth) == false){ inconsistency_found = true; return false; }
    mse_data* l = (mse_data*) left->get_data();
    mse_data* r = (mse_data*) right->get_data();

    if(l->occs.size() < SYMBOL_COUNT || r->occs.size() < SYMBOL_COUNT) return true;
    if(l->mean - r->mean > CHECK_PARAMETER){ inconsistency_found = true; return false; }
    if(r->mean - l->mean > CHECK_PARAMETER){ inconsistency_found = true; return false; }
    
    return true;
};

void mse_error::data_update_score(mse_data* l, mse_data* r){
    if(l->occs.size() <= STATE_COUNT || r->occs.size() <= STATE_COUNT) return;
    
    bool already_merged = false;

    total_merges = total_merges + 1;

    if(already_merged)
        num_points += r->occs.size();
    else
        num_points = num_points + l->occs.size() + r->occs.size();

    double error_left = 0.0;
    double error_right = 0.0;
    double error_total = 0.0;
    double mean_total = 0.0;
    
    mean_total = ((l->mean * ((double)l->occs.size()) + (r->mean * ((double)r->occs.size())))) / ((double)l->occs.size() + (double)r->occs.size());
    
    for(double_list::iterator it = l->occs.begin(); it != l->occs.end(); ++it){
        error_left  = error_left  + ((l->mean    - (double)*it)*(l->mean    - (double)*it));
        error_total = error_total + ((mean_total - (double)*it)*(mean_total - (double)*it));
    }
    for(double_list::iterator it = r->occs.begin(); it != r->occs.end(); ++it){
        error_right = error_right + ((r->mean    - (double)*it)*(r->mean    - (double)*it));
        error_total = error_total + ((mean_total - (double)*it)*(mean_total - (double)*it));
    }
    
    if(already_merged){
        RSS_before += error_right;
        RSS_after  += error_total - error_left;
    } else {
        RSS_before += error_right+error_left;
        RSS_after  += error_total;
    }
};

void mse_error::update_score(state_merger *merger, apta_node* left, apta_node* right){
    mse_data* l = (mse_data*) left->get_data();
    mse_data* r = (mse_data*) right->get_data();
    
    data_update_score(l, r);
};

double compute_RSS(apta_node* node){
    mse_data* l = (mse_data*) node->get_data();
    double error = 0.0;
    
    for(double_list::iterator it = l->occs.begin(); it != l->occs.end(); ++it){
        error  += ((l->mean    - (double)*it)*(l->mean    - (double)*it));
    }
    
    return error;
};

double mse_error::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    if(2*total_merges + num_points*(log(RSS_before/num_points)) - num_points*log(RSS_after/num_points) < 0) return -1;
    return 2*total_merges + num_points*(log(RSS_before/num_points)) - num_points*log(RSS_after/num_points);
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

bool is_low_occ_sink(apta_node* node){
    mse_data* l = (mse_data*) node->get_data();
    return l->occs.size() < STATE_COUNT;
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