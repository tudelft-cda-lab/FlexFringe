/**
 * @file types_state_comparator.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Compares the type and a hidden representation of two nodes. We currently use this 
 * one in active learning, when querying transformers. 
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include "parameters.h"
#include "types_state_comparator.h"
#include "input/inputdatalocator.h"

REGISTER_DEF_TYPE(type_state_comparator);
REGISTER_DEF_DATATYPE(type_state_comparator_data);

type_state_comparator_data::type_state_comparator_data() : evaluation_data() {
    total_paths = 0;
    total_final = 0;

    node_type = -1;
};

void type_state_comparator_data::initialize(){
    evaluation_data::initialize();
    total_paths = 0;
    total_final = 0;
    
    hidden_state.clear();
    node_type = -1;
};

void type_state_comparator_data::print_transition_label(iostream& output, int symbol){
    //output << "TODO" << endl;
};

void type_state_comparator_data::print_state_label(iostream& output){
    output << "TODO" << endl;
};

void type_state_comparator_data::print_transition_label_json(iostream& output, int symbol){
    //output << "TODO" << endl;
};

void type_state_comparator_data::print_state_label_json(iostream& output){
    output << "TODO" << endl;
};

void type_state_comparator_data::add_tail(tail* t){
    int type = t->get_type();
    if(node_type!=-1 && type != node_type) 
        throw logic_error("Input is not learnable. Found inconsisten types for the same path");

    if(!t->is_final()) {
        total_paths++;
    } else {
        total_final++;
    }
}

void type_state_comparator_data::del_tail(tail* t){
    //int type = t->get_type();
    if(!t->is_final()) {
        total_paths--;
    } else {
        total_final--;
    }
}

void type_state_comparator_data::read_json(json& data){
    total_final = data["total_final"];
    total_paths = data["total_paths"];
    node_type = data["node_type"];
};

void type_state_comparator_data::write_json(json& data){
    data["total_final"] = total_final;
    data["total_paths"] = total_paths;
    data["node_type"] = node_type;
};

void type_state_comparator_data::update(evaluation_data* right){
    auto* other = reinterpret_cast<type_state_comparator_data*>(right);

    total_paths += other->total_paths;
    total_final += other->total_final;
};

void type_state_comparator_data::undo(evaluation_data* right){
    auto* other = reinterpret_cast<type_state_comparator_data*>(right);

    total_paths -= other->total_paths;
    total_final -= other->total_final;
};


int type_state_comparator_data::predict_type(tail*){
    return node_type;
};

/**
 * @brief Computes cosine distance.
 * Cosine, because it is independent of the scaling factor of the vectors.
 */
double type_state_comparator::compute_state_distance(apta_node* left_node, apta_node* right_node){
    const auto& left_state = static_cast<type_state_comparator_data*>( left_node->get_data() )->get_state();
    const auto& right_state = static_cast<type_state_comparator_data*>( right_node->get_data() )->get_state();

    double denominator = 0;
    double left_square_sum = 0, right_square_sum = 0;
    for(int i=0; i<left_state.size(); i++){
        denominator += left_state[i] * right_state[i];

        left_square_sum += left_state[i] * left_state[i];
        right_square_sum += right_state[i] * right_state[i];
    }

    return denominator / ( sqrt(left_square_sum) * sqrt(right_square_sum) );
}

/**
 * @brief Compares type and hidden state.
 */
bool type_state_comparator::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){

    const static int mu = MU;

    if(inconsistency_found) return false;

    auto* l = (type_state_comparator_data*)left->get_data();
    auto* r = (type_state_comparator_data*)right->get_data();    
    if(l->get_type() != r->get_type()){
        inconsistency_found = true;
        return false;
    }

    auto state_distance = compute_state_distance(left, right);
    if(state_distance<mu){
        inconsistency_found = true;
        return false;       
    }

    return true;
};

void type_state_comparator::update_score(state_merger *merger, apta_node* left, apta_node* right){
	num_merges += 1;
};

double type_state_comparator::compute_score(state_merger *merger, apta_node* left, apta_node* right){
  return num_merges;
};

void type_state_comparator::reset(state_merger *merger){
  evaluation_function::reset(merger);
  num_merges = 0;
};

// ---------------------------------------------------------------- ---------------------------------
// ---------------------------------------------------------------- Sinks ----------------------------------------------------------------
// ---------------------------------------------------------------- ---------------------------------

// sinks for evaluation data type
/* bool type_state_comparator::is_low_count_sink(){
    return num_paths() + num_final() < SINK_COUNT;
}

int type_state_comparator::get_type_sink(){
    if(!USE_SINKS) return -1;
    return node_type;
}

bool type_state_comparator::sink_consistent(int type) {
    if (!USE_SINKS) return true;
    if (SINK_TYPE && get_type_sink() == type) return true;
    if (type == 0 && SINK_COUNT > 0 && is_low_count_sink()) return true;
    return false;
}

int type_state_comparator::num_sink_types(){
    if(!USE_SINKS) return 0;
    int result = 0;
    if(SINK_TYPE) result += inputdata_locator::get()->get_types_size();
    if(SINK_COUNT > 0) result += 1;
    return result;
}

int type_state_comparator::sink_type(){
    if(!USE_SINKS) return -1;
    if(SINK_TYPE){
        int result = get_type_sink();
        if(result != -1) return result + 1;
    }
    if(SINK_COUNT > 0 && is_low_count_sink()) return 0;
    return -1;
} */

