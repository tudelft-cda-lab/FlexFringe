#include <math.h>
#include <map>
#include "state_merger.h"
#include "evaluate.h"
#include "alergia.h"
#include "parameters.h"
#include "input/inputdatalocator.h"

REGISTER_DEF_DATATYPE(alergia_data);
REGISTER_DEF_TYPE(alergia);

/** Computing to get the divider with correction and paired pooling for merge testing */

void alergia::update_divider(double left_count, double right_count, double& left_divider, double& right_divider){
    //cerr << "updating " << left_count << " " << right_count << " " << left_divider << " " << right_divider << endl;
    if(left_count == 0.0 && right_count == 0.0) return;
    if(left_count < SYMBOL_COUNT || right_count < SYMBOL_COUNT) return;
    left_divider += left_count + CORRECTION;
    right_divider += right_count + CORRECTION;
    //cerr << "updating " << left_count << " " << right_count << " " << left_divider << " " << right_divider << endl;
}

void alergia::update_divider_pool(double left_pool, double right_pool, double& left_divider, double& right_divider){
    if(left_pool == 0.0 && right_pool == 0.0) return;
    left_divider += left_pool + CORRECTION;
    right_divider += right_pool + CORRECTION;
}

void alergia::update_left_pool(double left_count, double right_count, double& left_pool, double& right_pool){
    if(left_count == 0.0 && right_count == 0.0) return;
    if(right_count >= SYMBOL_COUNT) return;
    left_pool += left_count;
    right_pool += right_count;
}

void alergia::update_right_pool(double left_count, double right_count, double& left_pool, double& right_pool){
    if(left_count == 0.0 && right_count == 0.0) return;
    if(left_count >= SYMBOL_COUNT) return;
    left_pool += left_count;
    right_pool += right_count;
}

/** Computing the divider without pooling for prediction */

void alergia_data::get_symbol_divider(double& divider, double& count){
    divider = 0.0;
    double per_seen_correction = 0.0;
    for(auto & it : symbol_counts){
        if(it.second > 0){
            per_seen_correction += CORRECTION_PER_SEEN;
        }
    }
    if(FINAL_PROBABILITIES && num_final() > 0){
        per_seen_correction += CORRECTION_PER_SEEN;
    }
    for(auto & it : symbol_counts){
        if(it.second > 0) {
            divider += (double) it.second + CORRECTION + CORRECTION_SEEN + per_seen_correction;
        }
    }
    if(FINAL_PROBABILITIES && num_final() > 0){
        divider += (double) num_final() + CORRECTION + CORRECTION_SEEN + per_seen_correction;
    }

    count = (double)(CORRECTION + CORRECTION_UNSEEN) + per_seen_correction;
    divider += count;
}

void alergia_data::get_type_divider(double& divider, double& count){
    divider = 0.0;
    double per_seen_correction = 0.0;
    for(auto & it : path_counts){
        if(it.second > 0){
            per_seen_correction += CORRECTION_PER_SEEN;
        }
    }
    if(FINAL_PROBABILITIES && num_final() > 0){
        per_seen_correction += CORRECTION_PER_SEEN;
    }
    for(auto & it : path_counts){
        if(it.second > 0) {
            divider += (double) it.second + CORRECTION + CORRECTION_SEEN + per_seen_correction;
        }
    }
    if(FINAL_PROBABILITIES && num_final() > 0){
        for(auto & it : final_counts) {
            divider += (double) it.second + CORRECTION + CORRECTION_SEEN + per_seen_correction;
        }
    }

    count = (double)(CORRECTION + CORRECTION_UNSEEN) + per_seen_correction;
    divider += count;
}

/** Initialization and input reading/writing functions */

void alergia_data::initialize() {
    count_data::initialize();
    symbol_counts.clear();
}

void alergia_data::add_tail(tail* t){
    count_data::add_tail(t);

    if(t->is_final()) return;

    int symbol = t->get_symbol();
    if(symbol_counts.find(symbol) == symbol_counts.end()){
        symbol_counts[symbol] = 1;
    } else {
        symbol_counts[symbol]++;
    }
}

void alergia_data::del_tail(tail* t){
    count_data::del_tail(t);

    if(t->is_final()) return;

    int symbol = t->get_symbol();
    symbol_counts[symbol]--;
}

void alergia_data::read_json(json& data){
    count_data::read_json(data);

    json& d = data["trans_counts"];
    for (auto& symbol : d.items()){
        std::string sym = symbol.key();
        std::string val = symbol.value();
        symbol_counts[inputdata_locator::get()->symbol_from_string(sym)] = stoi(val);
    }
};

void alergia_data::write_json(json& data){
    count_data::write_json(data);

    data["symbol_counts"] = {};

    for(auto & symbol_count : symbol_counts) {
        int symbol = symbol_count.first;
        int value = symbol_count.second;
        data["trans_counts"][inputdata_locator::get()->string_from_symbol(symbol)] = std::to_string(value);
    }
};

void alergia_data::print_transition_label(std::iostream& output, int symbol){
    output << symbol_counts[symbol] << " ";
};

void alergia_data::print_state_label(std::iostream& output){
    count_data::print_state_label(output);
    output << "\n" << num_paths() << " " << num_final();
};

/** Merging update and undo_merge routines */

void alergia_data::update(evaluation_data* right){
    count_data::update(right);
    auto* other = (alergia_data*)right;

    for(auto & symbol_count : other->symbol_counts){
        symbol_counts[symbol_count.first] += symbol_count.second;
    }
};

void alergia_data::undo(evaluation_data* right){
    count_data::undo(right);
    auto* other = (alergia_data*)right;

    for(auto & symbol_count : other->symbol_counts){
        symbol_counts[symbol_count.first] -= symbol_count.second;
    }
};

/** Predict routines - only used during evaluation - not while learning
 * we do not pool, but do correct */

int alergia_data::predict_symbol(tail*){
    int max_count = -1;
    int max_symbol = 0;
    for(auto & symbol_count : symbol_counts){
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
    return max_symbol;
};

double alergia_data::predict_symbol_score(int t){
    double divider = 0.0;
    double count = 0.0;
    get_symbol_divider(divider, count);

    if(t != -1) {
        if (symbol_counts.find(t) != symbol_counts.end()) count += (double) symbol_counts[t];
    } else if (FINAL_PROBABILITIES) count += (double) num_final();
    else return 1.0;

    if(divider != 0.0) return log(count / divider);
    return 0.0;
}

double alergia_data::align_score(tail* t){
    return predict_score(t);
}

tail* alergia_data::sample_tail(){
    double divider = 0.0;
    double init_count = 0.0;
    get_symbol_divider(divider, init_count);

    tail* t = mem_store::create_tail(nullptr);
    t->td->symbol = -1;

    double rand_val = random_double();
    double sum_probs = 0.0;
    for(auto & it : symbol_counts){
        sum_probs += (init_count + it.second) / divider;
        if(sum_probs > rand_val){
            t->td->symbol = it.first;
            return t;
        }
    }
    return t;
}

/** computing the merge score and consistency */

double alergia::alergia_check(double right_count, double left_count, double right_total, double left_total){
    //cerr << "checking " << right_count << " " << left_count << " " << right_total << " " << left_total << endl;
    if(left_count == 0.0 && right_count == 0.0) return 0.0;
    if(left_total == 0.0 || right_total == 0.0) return 0.0;

    double bound = (1.0 / sqrt(left_total) + 1.0 / sqrt(right_total));
    bound = bound * sqrt(0.5 * log(2.0 / CHECK_PARAMETER));

    double gamma = ((left_count + CORRECTION) / left_total) - ((right_count + CORRECTION) / right_total);
    if(gamma < 0) gamma = -gamma;

    return bound - gamma;
}

bool alergia::alergia_test_and_update(double right_count, double left_count, double right_total, double left_total){
    double result = alergia_check(right_count, left_count, right_total, left_total);
    if(result < 0.0) return false;
    num_tests++;
    sum_diffs += result;
    return true;
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
    auto* l = (alergia_data*) left->get_data();
    auto* r = (alergia_data*) right->get_data();

    if(FINAL_PROBABILITIES){
        if(r->num_paths() + r->num_final() < STATE_COUNT || l->num_paths() + l->num_final() < STATE_COUNT) return true;
    } else {
        if(r->num_paths() < STATE_COUNT || l->num_paths() < STATE_COUNT) return true;
    }

    if(SYMBOL_DISTRIBUTIONS){
        if(!compute_tests(l->get_symbol_counts(), l->num_paths(), l->num_final(), r->get_symbol_counts(), r->num_paths(), r->num_final())){
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
