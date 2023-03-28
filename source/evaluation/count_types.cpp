#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include <map>
#include <set>
#include "parameters.h"
#include "count_types.h"
#include "input/inputdatalocator.h"

REGISTER_DEF_TYPE(count_driven);
REGISTER_DEF_DATATYPE(count_data);

count_data::count_data() : evaluation_data() {
    total_paths = 0;
    total_final = 0;
};

void count_data::initialize(){
    evaluation_data::initialize();
    total_paths = 0;
    total_final = 0;
    path_counts.clear();
    final_counts.clear();
};

void count_data::print_transition_label(iostream& output, int symbol){
};

void count_data::print_state_label(iostream& output){
    output << "fin: ";
    for(auto & final_count : final_counts){
        if(final_count.second > 0) output << inputdata_locator::get()->string_from_type(final_count.first) << ":" << final_count.second << " , ";
    }
    output << "\n path: ";
    for(auto & path_count : path_counts){
        if(path_count.second > 0) output << inputdata_locator::get()->string_from_type(path_count.first) << ":" << path_count.second << " , ";
    }
};

void count_data::print_transition_label_json(iostream& output, int symbol){
};

void count_data::print_state_label_json(iostream& output){
    output << "fin: ";
    for(auto & final_count : final_counts){
        output << final_count.first << ":" << final_count.second << " , ";
    }
    output << " path: ";
    for(auto & path_count : path_counts){
        output << path_count.first << ":" << path_count.second << " , ";
    }
};

void count_data::add_tail(tail* t){
    int type = t->get_type();
    if(!t->is_final()) {
        if(path_counts.find(type) == path_counts.end()){
            path_counts[type] = 1;
        } else {
            path_counts[type]++;
        }
        total_paths++;
    } else {
        if(final_counts.find(type) == final_counts.end()){
            final_counts[type] = 1;
        } else {
            final_counts[type]++;
        }
        total_final++;
    }
}

void count_data::del_tail(tail* t){
    int type = t->get_type();
    if(!t->is_final()) {
        path_counts[type]--;
        total_paths--;
    } else {
        final_counts[type]--;
        total_final--;
    }
}

void count_data::read_json(json& data){
    json d = data["final_counts"];
    for (auto& item : d.items()) {
        int type = stoi(item.key());
        int count = item.value();

        final_counts[type] = count;
    }

    d = data["path_counts"];
    for (auto& item : d.items()) {
        int type = stoi(item.key());
        int count = item.value();

        path_counts[type] = count;
    }

    total_final = data["total_final"];
    total_paths = data["total_paths"];
};

void count_data::write_json(json& data){
    for(auto & final_count : final_counts){
        int type = final_count.first;
        int count = final_count.second;

        data["final_counts"][to_string(type)] = count;
    }
    for(auto & path_count : path_counts) {
        int type = path_count.first;
        int count = path_count.second;

        data["path_counts"][to_string(type)] = count;
    }

    data["total_final"] = total_final;
    data["total_paths"] = total_paths;
};

void count_data::update(evaluation_data* right){
    auto* other = reinterpret_cast<count_data*>(right);
    for(auto & final_count : other->final_counts){
        int type = final_count.first;
        int count = final_count.second;
        if(final_counts.find(type) != final_counts.end()){
            final_counts[type] += count;
        } else {
            final_counts[type] = count;
        }
    }
    for(auto & path_count : other->path_counts){
        int type = path_count.first;
        int count = path_count.second;
        if(path_counts.find(type) != path_counts.end()){
            path_counts[type] += count;
        } else {
            path_counts[type] = count;
        }
    }
    total_paths += other->total_paths;
    total_final += other->total_final;
};

void count_data::undo(evaluation_data* right){
    auto* other = reinterpret_cast<count_data*>(right);

    for(auto & final_count : other->final_counts){
        int type = final_count.first;
        int count = final_count.second;
        final_counts[type] -= count;
    }
    for(auto & path_count : other->path_counts){
        int type = path_count.first;
        int count = path_count.second;
        path_counts[type] -= count;
    }
    total_paths -= other->total_paths;
    total_final -= other->total_final;
};

double count_data::predict_type_score(int t){
    double final_count = 0.0;
    double divider_correction = CORRECTION * (double)final_counts.size();
    if(divider_correction == 0.0) divider_correction += CORRECTION;
    if(final_counts.find(t) != final_counts.end() && final_counts[t] != 0.0) final_count = (double)final_counts[t] / (double)(total_final + divider_correction);
    final_count += (double)(CORRECTION) / (double)(total_final + divider_correction);

    if(!PREDICT_TYPE_PATH) return final_count;
    double path_count = CORRECTION;
    divider_correction = CORRECTION * (double)path_counts.size();
    if(divider_correction == 0.0) divider_correction += CORRECTION;
    if(path_counts.find(t) != path_counts.end() && path_counts[t] != 0.0) path_count = (double)path_counts[t] / (double)(total_paths+divider_correction);
    return (path_count + final_count) / 2.0;
};

int count_data::predict_type(tail*){
    int t = 0;
    double max_count = -1;
    for(int i = 0; i < inputdata_locator::get()->get_types_size(); ++i){ // -1 is the unknown type
        double prob = predict_type_score(i);
        if(max_count == -1 || max_count < prob){
            max_count = prob;
            t = i;
        }
    }
    return t;
};

/* default evaluation, count number of performed merges */
bool count_driven::consistent(state_merger *merger, apta_node* left, apta_node* right){
    if(inconsistency_found) return false;

    if(!TYPE_CONSISTENT) return true;
  
    auto* l = (count_data*)left->get_data();
    auto* r = (count_data*)right->get_data();

    for(auto & final_count : l->final_counts){
        int type = final_count.first;
        int count = final_count.second;
        if(count != 0){
            for(auto & final_count2 : r->final_counts){
                int type2 = final_count2.first;
                int count2 = final_count2.second;
                if(count2 != 0 && type2 != type){
                    inconsistency_found = true;
                    return false;
                }
            }
        }
    }
    
    return true;
};

void count_driven::update_score(state_merger *merger, apta_node* left, apta_node* right){
	num_merges += 1;
};

double count_driven::compute_score(state_merger *merger, apta_node* left, apta_node* right){
  return num_merges;
};

void count_driven::reset(state_merger *merger){
  num_merges = 0;
  evaluation_function::reset(merger);
  compute_before_merge=false;
};


// sinks for evaluation data type
bool count_data::is_low_count_sink(){
    return num_paths() + num_final() < SINK_COUNT;
}

int count_data::get_type_sink(){
    if(!USE_SINKS) return -1;

    int type = -1;
    for(auto & path_count : path_counts){
        int count = path_count.second;
        if(count != 0){
            if(type == -1) type = path_count.first;
            else if(type != path_count.first) return -1;
        }
    }
    for(auto & final_count : final_counts){
        int count = final_count.second;
        if(count != 0){
            if(type == -1) type = final_count.first;
            else if(type != final_count.first) return -1;
        }
    }
    return type;
}

bool count_data::sink_consistent(int type) {
    if (!USE_SINKS) return true;
    if (SINK_TYPE && get_type_sink() == type) return true;
    if (type == 0 && SINK_COUNT > 0 && is_low_count_sink()) return true;
    return false;
}

int count_data::num_sink_types(){
    if(!USE_SINKS) return 0;
    int result = 0;
    if(SINK_TYPE) result += inputdata_locator::get()->get_types_size();
    if(SINK_COUNT > 0) result += 1;
    return result;
}

int count_data::sink_type(){
    if(!USE_SINKS) return -1;
    if(SINK_TYPE){
        int result = get_type_sink();
        if(result != -1) return result + 1;
    }
    if(SINK_COUNT > 0 && is_low_count_sink()) return 0;
    return -1;
}

