#include "state_merger.h"
#include "evaluate.h"
#include "parameters.h"
#include "apta.h"

evaluation_data::evaluation_data(){
    node_type = -1;
    undo_pointer = 0;
    undo_consistent = false;
};

void evaluation_data::initialize(){
    node_type = -1;
    undo_pointer = 0;
    undo_consistent = false;
};

void evaluation_data::set_context(apta_node* n){
    node = n;
};

void evaluation_data::add_tail(tail* t){
};

void evaluation_data::del_tail(tail* t){
};

void evaluation_data::update(evaluation_data* right){
    if(node_type == -1){
        node_type = right->node_type;
        undo_pointer = right;
    }
};

void evaluation_data::undo(evaluation_data* right){
    if(right == undo_pointer){
        node_type = -1;
        undo_pointer = 0;
    }
};

void evaluation_data::split_update(evaluation_data* right){
    undo(right);
};

void evaluation_data::split_undo(evaluation_data* right){
    update(right);
};

bool evaluation_data::print_state_true(){
    return true;
};

void evaluation_data::print_state_label(iostream& output){
};

void evaluation_data::print_state_style(iostream& output){
};

void evaluation_data::print_transition_label(iostream& output, int symbol){
};

void evaluation_data::print_state_label_json(iostream& output){
};

void evaluation_data::print_state_style_json(iostream& output){
};

void evaluation_data::print_transition_label_json(iostream& output, int symbol){
};

// JSON output functions
void evaluation_data::print_state_properties(iostream& output) {
};

void evaluation_data::print_transition_properties(iostream& output, int symbol) {
};

// this should have a pair, set<pair<int, eval_data*>>
void evaluation_data::print_transition_style(iostream& output, set<int> symbols){
};

int evaluation_data::sink_type(){
    return -1;
};

bool evaluation_data::sink_consistent(int type){   
    return true;
};

void evaluation_data::write_json(json& node){
}

void evaluation_data::read_json(json& node){
}

int evaluation_data::predict_type(tail*){
    return 0;
};

double evaluation_data::predict_type_score(int t){
    return 0.0;
};

double evaluation_data::predict_type_score(tail* t){
    return predict_type_score(t->get_type());
};

int evaluation_data::predict_symbol(tail*){
    return 0;
};

double evaluation_data::predict_symbol_score(int s){
    return 0.0;
};

double evaluation_data::predict_symbol_score(tail* t){
    return predict_symbol_score(t->get_symbol());
};

double evaluation_data::predict_attr(tail*, int attr){
    return 0.0;
};

double evaluation_data::predict_attr_score(int attr, double v) {
    return 0.0;
};

double evaluation_data::predict_attr_score(int attr, tail* t){
    return predict_attr_score(attr, t->get_symbol_value(attr));
};

string evaluation_data::predict_data(tail*){
    return "0";
};

double evaluation_data::predict_data_score(string s){
    return 0.0;
};

double evaluation_data::predict_data_score(tail* t){
    return predict_data_score(t->get_data());
};

double evaluation_data::predict_score(tail* t){
    return predict_symbol_score(t);
};

double evaluation_data::align_score(tail* t){
    if(node->child(t->get_symbol()) != nullptr) return 0.0;
    return -1;
};

bool evaluation_data::align_consistent(tail* t){
    if(t->is_final()) return false;
    if(node->child(t) == nullptr) return false;
    return true;
};

tail* evaluation_data::sample_tail() {
    return mem_store::create_tail(nullptr);
}

const float evaluation_data::get_weight(const int symbol) const {
    cerr << "WARNING: get_weight() method not implemented for the heuristic you are using. Perhaps you
    chose the wrong counterexample search strategy?" << endl;
    throw std::exception();
}


/* defa */ 
evaluation_function::evaluation_function() {
    compute_before_merge = false;
};

void evaluation_function::set_params(string params) {   
  this->evalpar = params;
};

bool evaluation_function::merge_no_root(apta_node* left, apta_node* right){
    if(left->get_source() == 0) {inconsistency_found = true; return false;};
    return true;
}

bool evaluation_function::merge_same_depth(apta_node* left, apta_node* right){
    if(left->get_depth() != right->get_depth()) {inconsistency_found = true; return false;};
    return true;
}

bool evaluation_function::merge_no_final(apta_node* left, apta_node* right){
    if(left->get_final() != 0 && right->get_final() == 0) {inconsistency_found = true; return false;};
    if(left->get_final() == 0 && right->get_final() != 0) {inconsistency_found = true; return false;};
    return true;
}

int evaluation_function::merge_depth_score(apta_node* left, apta_node* right){
    set<apta_node*> path;
    for(apta_node* n = left; n != 0; n = n->get_source()->find()){
        path.insert(n);
        if(n->get_source() == 0) break;
    }
    for(apta_node* n = right; n != 0; n = n->get_source()->find()){
        if(path.find(n) != path.end()){
            return n->get_depth();
        }
        if(n->get_source() == 0) break;
    }
    return 0;
};

bool evaluation_function::pre_consistent(state_merger *merger, apta_node* left, apta_node* right){
    return true;
};

bool evaluation_function::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
  if(inconsistency_found) return false;
  
  if(left->get_data()->node_type != -1 && right->get_data()->node_type != -1 && left->get_data()->node_type != right->get_data()->node_type){
      inconsistency_found = true;
      return false;
  }
    
  return true;
};

void evaluation_function::update_score(state_merger *merger, apta_node* left, apta_node* right){
    num_merges += 1;
    merged_left_states.insert(left);
};

void evaluation_function::update_score_after(state_merger *merger, apta_node* left, apta_node* right){
};

void evaluation_function::update_score_after_recursion(state_merger *merger, apta_node* left, apta_node* right){
};

bool evaluation_function::split_consistent(state_merger *merger, apta_node* left, apta_node* right){
    return true;
};

void evaluation_function::split_update_score_before(state_merger *merger, apta_node* left, apta_node* right, tail* t){
    evaluation_data* r = right->get_data();

    if(right->get_size() == 0) num_merges += 1;

    if(r->undo_consistent) num_inconsistencies -= 1;
    r->undo_consistent = false;
};

void evaluation_function::split_update_score_after(state_merger *merger, apta_node* left, apta_node* right, tail* t){
    evaluation_data* r = right->get_data();

    if(left->get_size() == 0) num_merges -= 1;

    inconsistency_found = false;
    consistent(merger, left, right, 0); // TODO: how to set depth here?
    if(inconsistency_found){
        r->undo_consistent = true;
        num_inconsistencies += 1;
    }
};

void evaluation_function::split_update_score_before(state_merger*, apta_node* left, apta_node* right){
    return;
};

void evaluation_function::split_update_score_after(state_merger*, apta_node* left, apta_node* right) {
    num_merges -= 1;
    return;
};
 
bool evaluation_function::compute_consistency(state_merger *merger, apta_node* left, apta_node* right){
  return inconsistency_found == false;
};

double evaluation_function::compute_score(state_merger *merger, apta_node* left, apta_node* right){
  return num_merges;
};

bool evaluation_function::split_compute_consistency(state_merger *merger, apta_node* left, apta_node* right){
    return num_inconsistencies != 0;
};

double evaluation_function::split_compute_score(state_merger *merger, apta_node* left, apta_node* right){
    return ((double)num_inconsistencies) / ((double)num_merges);
};

bool evaluation_function::sink_convert_consistency(state_merger *merger, apta_node* left, apta_node* right){
    if(right->get_source() != 0 && right->get_source()->find() == left) return true;
    if(right->get_source() != 0 && left->get_source() != 0 && left->get_source()->find() == right->get_source()->find()) return true;
    return false;
};

void evaluation_function::reset(state_merger *merger){
  inconsistency_found = false;
  num_merges = 0;
  merged_left_states.clear();
  num_inconsistencies = 0;
};

void evaluation_function::reset_split(state_merger *merger, apta_node *node){
    reset(merger);
};

void evaluation_function::update(state_merger *merger){
};

void evaluation_function::initialize_after_adding_traces(state_merger *merger){
};
void evaluation_function::initialize_before_adding_traces(){
};

/** When is an APTA node a sink state?
 * sink states are not considered merge candidates
 *								*/
bool is_low_count_sink(apta_node* node){
    node = node->find();
    return node->get_size() < SINK_COUNT;
}

bool evaluation_data::sink_consistent(apta_node* node, int type){
    if(!USE_SINKS) return true;
    if(type == 0) return is_low_count_sink(node);
    return true;
};

int evaluation_data::num_sink_types(){
    if(!USE_SINKS) return 0;
    return 2;
};

double evaluation_function::compute_global_score(state_merger* merger) {
    return (double)merger->get_final_apta_size();
};

double evaluation_function::compute_partial_score(state_merger* merger) {
    return (double)merger->get_num_red_states();
};

void evaluation_function::set_context(state_merger* m){
    merger = m;
}
