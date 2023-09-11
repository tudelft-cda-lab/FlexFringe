/**
 * @file log_alergia.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <math.h>
#include <map>
#include "state_merger.h"
#include "evaluate.h"
#include "log_alergia.h"
#include "parameters.h"
#include "input/inputdatalocator.h"

REGISTER_DEF_DATATYPE(log_log_alergia_data);
REGISTER_DEF_TYPE(log_alergia);

/** Initialization and input reading/writing functions */

void log_alergia_data::initialize() {
    evaluation_data::initialize();
    //symbol_pro.clear();
}

void log_alergia_data::add_tail(tail* t){
    evaluation_data::add_tail(t);
}

void log_alergia_data::del_tail(tail* t){
    evaluation_data::del_tail(t);
}

void log_alergia_data::read_json(json& data){
    evaluation_data::read_json(data);

    json& d = data["trans_probs"];
    for (auto& symbol : d.items()){
        string sym = symbol.key();
        string val = symbol.value();
        symbol_probability_map[inputdata_locator::get()->symbol_from_string(sym)] = stod(val);
    }
};

void log_alergia_data::write_json(json& data){
    evaluation_data::write_json(data);

    data["symbol_probs"] = {};

    for(auto & symbol_count : symbol_probability_map) {
        int symbol = symbol_count.first;
        double value = symbol_count.second;
        data["trans_probs"][inputdata_locator::get()->string_from_symbol(symbol)] = to_string(value);
    }
};

void log_alergia_data::print_transition_label(iostream& output, int symbol){
    output << symbol_probability_map[symbol] << ": ";
};

void log_alergia_data::print_state_label(iostream& output){
    evaluation_data::print_state_label(output);
    //output << "\n" << num_paths() << " " << num_final();
};

/** Merging update and undo_merge routines */

void log_alergia_data::update(evaluation_data* right){
    evaluation_data::update(right);
    auto* other = (log_alergia_data*)right;

    for(auto & symbol_p_mapping : other->symbol_probability_map){
        auto& p1 = symbol_probability_map[symbol_p_mapping.first];
        auto& p2 = symbol_p_mapping.second;
        symbol_probability_map[symbol_p_mapping.first] = (p1 + p2) / static_cast<double>(2);
    }
};

void log_alergia_data::undo(evaluation_data* right){
    evaluation_data::undo(right);
    auto* other = (log_alergia_data*)right;

    for(auto & symbol_count : other->symbol_probability_map){
        auto& p1 = symbol_probability_map[symbol_p_mapping.first];
        auto& p2 = symbol_p_mapping.second;

        symbol_probability_map[symbol_p_mapping.first] = p1 * static_cast<double>(2) - p2; // reverse update-operation
    }
};

/**
 * @brief 
 * 
 * @return int 
 */
int log_alergia_data::predict_symbol(tail*){
    // TODO
    /* int max_count = -1;
    int max_symbol = 0;
    for(auto & symbol_count : symbol_probability_map){
        int count = symbol_count.second;
        if(max_count == -1 || max_count < count){
            max_count = count;
            max_symbol = symbol_count.first;
        }
    }
    if(FINAL_PROBABILITIES){
        if(max_count == -1 || max_count < num_final()){
            max_symbol = -1;
        }
    }
    return max_symbol; */
};

double log_alergia_data::predict_symbol_score(int t){
    if(t == -1) {
        1.0;
    } 

    if(symbol_probability_map.contains(t->get_symbol()))
        return symbol_probability_map[t->get_symbol()];
    
    return 0.0;
}

double log_alergia_data::align_score(tail* t){
    return predict_score(t);
}


bool alergia::compute_tests(num_map& left_map, int left_total, int left_final,
                            num_map& right_map, int right_total, int right_final){

    /* computing the dividers (denominator) */
    double left_divider = 0.0; double right_divider = 0.0;
    /* we pool low frequency counts (sum them up in a separate bin), decided by input parameter SYMBOL_COUNT
    * we create pools 1 and 2 separately for left and right low counts
    * in this way, we can detect differences in distributions even if all counts are low (i.e. [0,0,1,1] vs [1,1,0,0]) */
    double l1_pool = 0.0; double r1_pool = 0.0; double l2_pool = 0.0; double r2_pool = 0.0;

    /*
    for(auto & it : left_map){
        cerr << it.first << " : " << it.second << " , ";
    }
    cerr << endl;
    for(auto & it : right_map) {
        cerr << it.first << " : " << it.second << " , ";
    }
    cerr << endl;
    */


    int matching_right = 0;
    for(auto & it : left_map){
        int type = it.first;
        double left_count = it.second;
        if(left_count == 0) continue;

        int right_count = 0.0;
        auto hit = right_map.find(type);
        if(hit != right_map.end()) right_count = hit->second;
        matching_right += right_count;

        update_divider(left_count, right_count, left_divider, right_divider);
        update_left_pool(left_count, right_count, l1_pool, r1_pool);
        update_right_pool(left_count, right_count, l2_pool, r2_pool);
    }
    r2_pool += right_total - matching_right;

    /* optionally add final probabilities (input parameter) */
    if(FINAL_PROBABILITIES){
        update_divider(left_final, right_final, left_divider, right_divider);
        update_left_pool(left_final, right_final, l1_pool, r1_pool);
        update_right_pool(left_final, right_final, l2_pool, r2_pool);
    }

    update_divider_pool(l1_pool, r1_pool, left_divider, right_divider);
    update_divider_pool(l2_pool, r2_pool, left_divider, right_divider);

    if((l1_pool != 0 || r1_pool != 0) && !alergia_test_and_update(l1_pool, r1_pool, left_divider, right_divider)){
        return false;
    }
    if((l2_pool != 0 || r2_pool != 0) && !alergia_test_and_update(l2_pool, r2_pool, left_divider, right_divider)){
        return false;
    }

    /* we have calculated the dividers and pools */
    for(auto & it : left_map){
        int type = it.first;
        double left_count = it.second;
        if(left_count == 0) continue;

        int right_count = 0.0;
        auto hit = right_map.find(type);
        if(hit != right_map.end()) right_count = hit->second;
        matching_right += right_count;

        if(!alergia_test_and_update(left_count, right_count, left_divider, right_divider)){
            return false;
        }
    }
    return true;
}

/* ALERGIA, consistency based on Hoeffding bound, only uses positive (type=1) data, pools infrequent counts */
bool alergia::consistent(state_merger *merger, apta_node* left, apta_node* right){
    //if(inconsistency_found) return false;
    auto* l = (log_alergia_data*) left->get_data();
    auto* r = (log_alergia_data*) right->get_data();

    if(FINAL_PROBABILITIES){
        if(r->num_paths() + r->num_final() < STATE_COUNT || l->num_paths() + l->num_final() < STATE_COUNT) return true;
    } else {
        if(r->num_paths() < STATE_COUNT || l->num_paths() < STATE_COUNT) return true;
    }

    if(SYMBOL_DISTRIBUTIONS){
        if(!compute_tests(l->get_symbol_probability_map(), l->num_paths(), l->num_final(), r->get_symbol_probability_map(), r->num_paths(), r->num_final())){
            inconsistency_found = true; return false;
        }
    }
    if(TYPE_DISTRIBUTIONS){
        if(!compute_tests(l->path_counts, l->num_paths(), 0, r->path_counts, r->num_paths(), 0)){
            inconsistency_found = true; return false;
        }
        if(!compute_tests(l->final_counts, l->num_final(), 0, r->final_counts, r->num_final(), 0)){
            inconsistency_found = true; return false;
        }
    }
    return true;
};

double alergia::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    return sum_diffs;
};

void alergia::reset(state_merger *merger){
    inconsistency_found = false;
    sum_diffs = 0.0;
    num_tests = 0.0;
};
