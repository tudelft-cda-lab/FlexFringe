#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include <map>

#include "parameters.h"
#include "mealy.h"

REGISTER_DEF_TYPE(mealy);
REGISTER_DEF_DATATYPE(mealy_data);

int mealy_data::num_outputs;
si_map mealy_data::output_int;
is_map mealy_data::int_output;

void mealy_data::add_tail(tail* t){
    evaluation_data::add_tail(t);
    if(output_int.find(t->get_data()) == output_int.end()){
       output_int[t->get_data()] = num_outputs;
       int_output[num_outputs] = t->get_data();
       num_outputs++;
    }
    outputs[t->get_symbol()] = output_int[t->get_data()];
};

void mealy_data::update(evaluation_data* right){
    mealy_data* other = reinterpret_cast<mealy_data*>(right);
    
    for(output_map::iterator it = other->outputs.begin(); it != other->outputs.end(); ++it){
        int input  = it->first;
        int output = it->second;
        
        if(outputs.find(input) == outputs.end()){
            outputs[input] = output;
            undo_info[input] = other;
        }
    }
};

void mealy_data::undo(evaluation_data* right){
    mealy_data* other = reinterpret_cast<mealy_data*>(right);

    for(output_map::iterator it = other->outputs.begin(); it != other->outputs.end(); ++it){
        int input  = it->first;
        //int output = it->second;
        
        undo_map::iterator it2 = undo_info.find(input);
        
        if(it2 != undo_info.end() && it2->second == other){
            outputs.erase(input);
            undo_info.erase(input);
        }
    }
};

/* default evaluation, count number of performed merges */
bool mealy::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    if(inconsistency_found) return false;
  
    mealy_data* l = reinterpret_cast<mealy_data*>(left->get_data());
    mealy_data* r = reinterpret_cast<mealy_data*>(right->get_data());
    
    int matched = 0;
    
    for(output_map::iterator it = r->outputs.begin(); it != r->outputs.end(); ++it){
        int input  = it->first;
        int output = it->second;

        if(l->outputs.find(input) != l->outputs.end()){
            if(l->outputs[input] != output){
                inconsistency_found = true;
                return false;
            }
            matched = matched + 1;
        }
    }
    
    num_unmatched = num_unmatched + (l->outputs.size() - matched);
    num_unmatched = num_unmatched + (r->outputs.size() - matched);
    num_matched   = num_matched   + matched;
    
    return true;
};

bool mealy::compute_consistency(state_merger* merger, apta_node* left, apta_node* right){
    if(evaluation_function::compute_consistency(merger, left, right) == false) return false;
    return true;
};

double mealy::compute_score(state_merger* merger, apta_node* left, apta_node* right){
    return num_matched;
};

void mealy::reset(state_merger* merger){
    evaluation_function::reset(merger);
    num_matched = 0;
    num_unmatched = 0;
};


void mealy_data::print_transition_label(iostream& output, int symbol){
   output << int_output[outputs[symbol]];
};

string mealy_data::predict_data(tail* t){
    if(t == nullptr) return "0";
    return int_output[outputs[t->get_symbol()]];
};