#include "state_merger.h"
#include "evaluate.h"
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <cstdio>

#include "parameters.h"
#include "dfa_properties.h"
#include "input/inputdatalocator.h"

/**
 * @brief Construct a new state merger::state merger object.
 * 
 * @param e Pointer to the evaluation function.
 * @param a Pointer to the augmented prefix tree acceptor.
 */
state_merger::state_merger(inputdata* d, evaluation_function* e, apta* a){
    aut = a;
    eval = e;
    dat = d;

    left_depth_map = nullptr;
    right_depth_map = nullptr;
}

/**
 * @brief TODO
 * 
 * @param t 
 * @return apta_node* 
 */
apta_node* state_merger::get_state_from_trace(trace* t) const{
    tail* cur_tail = t->head;
    apta_node* cur_state = aut->root;
    while(cur_tail != nullptr){
        cur_state = cur_state->find();
        cur_state = cur_state->child(cur_tail);
        cur_tail = cur_tail->future();
    }
    return cur_state;
}

/**
 * @brief TODO
 * 
 * @param n 
 * @return tail* 
 */
trace* state_merger::get_trace_from_state(apta_node* n){
    return n->access_trace;
}

/* --------------------------------- Special state sets, used by red blue framework -----------------------------*/

/**
 * @brief Get the candidate states used by SAT encoding.
 * red and blue sets can be accessed directly
 * 
 * @return state_set& 
 */
state_set* state_merger::get_all_states() const{
    auto* states = new state_set();
    for(merged_APTA_iterator Ait = merged_APTA_iterator(aut->root); *Ait != nullptr; ++Ait){
            states->insert(*Ait);
    }
    return states;
}

bool is_sink_node(apta_node* node){
    return node->get_data()->sink_type() != -1;
}

state_set* state_merger::get_red_states() const{
    auto* states = new state_set();
    for(red_state_iterator Ait = red_state_iterator(aut->root); *Ait != nullptr; ++Ait){
        states->insert(*Ait);
    }
    return states;
}

state_set* state_merger::get_blue_states() const{
    auto* states = new state_set();
    for(blue_state_iterator Ait = blue_state_iterator(aut->root); *Ait != nullptr; ++Ait){
        states->insert(*Ait);
    }
    return states;
}

state_set* state_merger::get_non_sink_states() const{
    auto* states = new state_set();
    for(merged_APTA_iterator_func Ait = merged_APTA_iterator_func(aut->root, is_sink_node); *Ait != nullptr; ++Ait){
        states->insert(*Ait);
    }
    return states;
}

state_set* state_merger::get_candidate_states(){
    auto* states = get_blue_states();
    auto* candidate_states = new state_set();
    for(auto state : *states){
        if(sink_type(state) == -1) {
            for (merged_APTA_iterator Ait = merged_APTA_iterator(state); *Ait != nullptr; ++Ait) {
                candidate_states->insert(*Ait);
            }
        }
    }
    delete states;
    return candidate_states;
}

/**
 * @brief Obtain the sink states.
 * 
 * @return state_set& The sink states as a reference.
 */
state_set* state_merger::get_sink_states(){
    auto* states = get_blue_states();
    auto* sink_states = new state_set();
    for(auto state : *states){
        if(sink_type(state) != -1){
            for(merged_APTA_iterator Ait = merged_APTA_iterator(state); *Ait != nullptr; ++Ait){
                sink_states->insert(*Ait);
            }
        }
    }
    delete states;
    return sink_states;
}

/**
 * @brief Get the size of the final apta.
 * 
 * @return int The size of the final apta.
 */
int state_merger::get_final_apta_size() const{
    int result = 0;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(aut->root); *Ait != nullptr; ++Ait){
        result++;
    }
    return result;
}

int state_merger::get_num_red_states() const{
    int result = 0;
    for(red_state_iterator Ait = red_state_iterator(aut->root); *Ait != nullptr; ++Ait){
        result++;
    }
    return result;
}

int state_merger::get_num_red_transitions() const{
    int result = 0;
    for(red_state_iterator Ait = red_state_iterator(aut->root); *Ait != nullptr; ++Ait){
        for (auto & guard : (*Ait)->guards) {
            apta_node *child = guard.second->target;
            if(child != nullptr && child->red) result++;
        }
    }
    return result;
}
/* --------------------------------- End of Special state sets, used by red blue framework -----------------------------*/

/*  */

/* --------------------------------- BEGIN basic state merging routines, these can be accessed directly, for instance to compute a conflict graph
   the search routines do not access these methods directly, but use the perform and test merge routines below -----------------------------*/

/* standard merge, the process is interupted when an inconsistency is found */
void state_merger::pre_split(apta_node* left, apta_node* right, int depth, bool evaluate, bool perform, bool test){
    if(left->performed_splits == nullptr) return;
    for(auto it = left->performed_splits->rbegin(); it != left->performed_splits->rend(); ++it){
        split_init(right, it->first, it->second, depth, evaluate, true, test);
    }
}

void state_merger::undo_pre_split(apta_node* left, apta_node* right){
    if(right->performed_splits == nullptr) return;
    while(!right->performed_splits->empty()){
        auto it = right->performed_splits->begin();
        undo_split_init(right, it->first, it->second);
    }
}

bool state_merger::early_stop_merge(apta_node* left, apta_node* right, int depth, bool& keep_merging){
    keep_merging = true;
    if (KTAIL != -1 && depth > KTAIL) return true;
    if (KSTATE != -1 && (left->size < KSTATE || right->size < KSTATE)) return true;

    keep_merging = false;
    if(left->red && STAR_FREE && counting_path_occurs(left, right)) return true;
    if ((left->red && RED_FIXED) || ALL_FIXED) {
        for (auto & guard : right->guards) {
            apta_node *child = guard.second->target;
            if (child == nullptr) continue;
            if (left->child(guard.first) == nullptr) return true;
        }
    }

    return false;
}

bool state_merger::merge(apta_node* left, apta_node* right, int depth, bool evaluate, bool perform, bool test){
    if(left == nullptr || right == nullptr) return true;

    bool result = true;

    if(test){
        bool early_stop = false;

        if(early_stop_merge(left, right, depth, early_stop)) return early_stop;
        if(evaluate && !eval->consistent(this, left, right)) return false;
    }
    
    if(perform){
        pre_split(left, right, depth, evaluate, perform, test);
    }
    if(evaluate){
        eval->update_score(this, left, right);
    }
    if(perform){
        right->merge_with(left);
        if(MERGE_DATA) left->data->update(right->data);
    }
    if(evaluate) {
        eval->update_score_after(this, left, right);
    }
    for(auto & it : right->guards){
        if(it.second->target == nullptr) continue;
        int i = it.first;
        apta_guard* right_guard = it.second;
        apta_guard* left_guard = left->guard(i, it.second);
        
        if(left_guard == nullptr || left_guard->target == nullptr){
            if(perform){
                left->set_child(i, right_guard->target);
            }
        } else {
            apta_node* right_child = right_guard->target;
            apta_node* left_child = left_guard->target;
            
            apta_node* child = left_child->find();
            apta_node* other_child = right_child->find();
            
            if(child != other_child){
                result = merge(child, other_child, depth + 1, evaluate, perform, test);
                if(test && !result) break;
            }
        }
    }
    if(evaluate) {
        eval->update_score_after_recursion(this, left, right);
    }
    return true;
}

bool state_merger::merge(apta_node* left, apta_node* right) {
    return merge(left, right, 0, true, true, true);
}

/* forced merge, continued even when inconsistent */
void state_merger::merge_force(apta_node* left, apta_node* right){
    merge(left, right, 0, false, true, false);
}

/* testing merge, no actual merge is performed, only consistency and score are computed */
bool state_merger::merge_test(apta_node* left, apta_node* right){
    return merge(left, right, 0, true, false, true);
}

/* undo_merge merge, works for both forcing and standard merging, not needed for testing merge */
void state_merger::undo_merge(apta_node* left, apta_node* right){
    if(left == nullptr || right == nullptr) return;
    if(right->representative == nullptr) return;
    assert(left != right);
    assert(right->representative == left);
    
    for(auto it = right->guards.rbegin();it != right->guards.rend(); ++it){
        if(it->second->target == nullptr) continue;
        int i = it->first;
        apta_guard* right_guard = it->second;
        apta_guard* left_guard = left->guard(i, it->second);
        if(left_guard == nullptr) continue;
        
        apta_node* right_child = right_guard->target;
        apta_node* left_child = left_guard->target;

        if(left_child == right_child){
            left_guard->target = nullptr;
        } else if(left_child != nullptr) {
            left_child = left_child->find();
            right_child = right_child->find_until(left_child);
            undo_merge(left_child, right_child);
        }
    }

    right->undo_merge_with(left);
    if(MERGE_DATA) left->data->undo(right->data);
    undo_pre_split(left, right);
}

/* END OF STATE MERGING ROUTINES */

bool state_merger::split_single(apta_node* new_node, apta_node* old_node, tail* t, int depth, bool evaluate, bool perform, bool test){
    if(t->future() == nullptr) return true;

    if(test) {
        // test early stopping conditions
        if (KTAIL != -1 && depth > KTAIL) return true;
        if (KSTATE != -1 && (new_node->size < KSTATE && old_node->size < KSTATE)) return true;
    }

    int symbol = t->get_symbol();
    if(symbol == -1) return true;
    apta_node* old_child = old_node->child(symbol)->find();
    apta_node* new_child = new_node->child(symbol);
    if(new_child == nullptr){
        new_child = mem_store::create_node(old_child);
        temporary_node_store.push_front(new_child);
        new_node->set_child(symbol,new_child);
        new_child->source = new_node;
        new_child->depth = new_node->depth + 1;
    }

    if(evaluate) eval->split_update_score_before(this, old_child, new_child, t->future());
    new_child->size++;
    old_child->size--;

    tail* future_tail = t->future();
    
    old_child->data->del_tail(future_tail);
    tail* new_tail = mem_store::create_tail(future_tail);
    future_tail->split(new_tail);
    new_child->add_tail(new_tail);
    new_child->data->add_tail(new_tail);    
    split_single(new_child, old_child, new_tail, depth+1, evaluate, perform, test);
    
    if(evaluate) eval->split_update_score_after(this, old_child, new_child, t->future());

    return true;
}

void state_merger::undo_split_single(apta_node* new_node, apta_node* old_node, tail* t){
    assert(t != nullptr);

    int symbol = t->get_symbol();
    apta_node* old_child = old_node->child(symbol);
    apta_guard* new_guard = new_node->guard(symbol);
    apta_node* new_child = new_node->child(symbol);

    if(new_child != nullptr && old_child != nullptr) {
        old_child = old_child->find();
        undo_split_single(new_child, old_child, t->future());
        new_child->size--;
        old_child->size++;

        if (new_child->size == 0) {
            new_guard->target = nullptr;
            mem_store::delete_tail(new_child->tails_head);
            new_child->tails_head = nullptr;
            mem_store::delete_node(new_child);

            old_node->data->split_undo(new_node->data);
        }
    }
    t->split_from->undo_split();
}

void state_merger::undo_split_single(apta_node* new_node, apta_node* old_node){
    for(auto & guard : new_node->guards){
        if(guard.second->target == nullptr) continue;
        int i = guard.first;
        apta_guard* new_guard = guard.second;
        apta_guard* old_guard = old_node->guard(i, guard.second);
        
        apta_node* new_child = new_guard->target;
        apta_node* old_child = old_guard->target->find();
        
        undo_split_single(new_child, old_child);

        new_guard->target = nullptr;
    }
    
    old_node->size = old_node->size + new_node->size;
    for(tail* t = new_node->tails_head; t != nullptr; t = t->next()){
        t->split_from->undo_split();
    }
    
    old_node->data->split_undo(new_node->data);
}

/* new_node and old_node are already split
 * this function performs the split recursively on their children */
bool state_merger::split(apta_node* new_node, apta_node* old_node, int depth, bool evaluate, bool perform, bool test){
    if(new_node->tails_head == nullptr) return true;

    tail_iterator it = tail_iterator(old_node);
    tail* t = *it;
    new_node->access_trace = inputdata_locator::get()->access_trace(new_node->tails_head->past());
    if(t != nullptr) old_node->access_trace = inputdata_locator::get()->access_trace(t->past());

    if(test) {
        // test early stopping conditions
        if (KTAIL != -1 && depth > KTAIL) return true;
        if (KSTATE != -1 && (new_node->size < KSTATE && old_node->size < KSTATE)) return true;
    }

    /*
    for (auto it = old_node->guards.begin(); it != old_node->guards.end(); ++it) {
        if(it->second->target == nullptr) continue;
        if(it->second->target->size == 0) continue;
        int symbol = it->first;
        apta_node *old_child = it->second->target->find();
        apta_node *new_child = new_node->child(symbol);
        if (new_child == nullptr) {
            new_child = mem_store::create_node(old_child);
            new_node->set_child(symbol, new_child);
            new_child->source = new_node;
            new_child->depth = new_node->depth + 1;
        }
    }
    */

    for (tail *t = new_node->tails_head; t != nullptr; t = t->next()) {
        if (t->get_index() == -1) continue;
        if (t->future() == nullptr) continue;

        int symbol = t->get_symbol();
        apta_node *old_child = old_node->child(symbol)->find();
        apta_node *new_child = new_node->child(symbol);
        if (new_child == nullptr) {
            new_child = mem_store::create_node(old_child);
            new_node->set_child(symbol, new_child);
            new_child->source = new_node;
            new_child->depth = new_node->depth + 1;
        }
        new_child->size++;
        tail* future_tail = t->future();
        tail *new_tail = mem_store::create_tail(future_tail);
        future_tail->split(new_tail);
        new_child->add_tail(new_tail);
        new_child->data->add_tail(new_tail);
    }

    for (auto it = new_node->guards.begin(); it != new_node->guards.end(); ++it) {
        if (it->second->target == nullptr) continue;
        if (it->second->target->size == 0) continue;

        apta_guard *new_guard = it->second;
        apta_guard *old_guard = old_node->guard(it->first);

        apta_node* new_child = new_guard->target;
        apta_node* old_child = old_guard->target->find();

        if(new_child->size == old_child->size){
            for (tail *t = new_child->tails_head; t != nullptr; t = t->next()) {
                t->split_from->undo_split();
            }
            new_guard->target = nullptr;
            new_guard->undo_split = new_child;
            if(perform) {
                new_guard->target = old_child;
                old_guard->target = nullptr;

                if (old_child->original_source == nullptr) old_child->original_source = old_child->source;
                old_child->source = new_node;
            }
        } else {
            old_child->size -= new_child->size;
            if(evaluate){
                eval->split_update_score_before(this, old_child, new_child);
            }
            if(perform){
                old_child->data->split_update(new_child->data);
            }
            if(evaluate){
                eval->split_update_score_after(this, old_child, new_child);
            }

            split(new_child, old_child, depth+1, evaluate, perform, test);
        }
    }
    return true;
}

/* new_node and old_node are not yet unsplit
 * this function first performs the unsplit recursively on their children */
void state_merger::undo_split(apta_node* new_node, apta_node* old_node){
    for(auto it = new_node->guards.rbegin(); it != new_node->guards.rend(); ++it){
        if(it->second->target == nullptr) continue;
        int i = it->first;
        apta_guard* new_guard = it->second;
        apta_node *new_child = new_guard->target;
        apta_guard* old_guard = old_node->guard(i, it->second);

        if(old_guard->target == nullptr){
            assert(new_guard->undo_split != nullptr);

            old_guard->target = new_child;
            new_guard->target = nullptr;
            if(new_child->original_source->find() == old_node){
                new_child->source = new_child->original_source;
            } else {
                new_child->source = old_node;
            }

            mem_store::delete_tail(new_guard->undo_split->tails_head);
            new_guard->undo_split->tails_head = nullptr;
            mem_store::delete_node(new_guard->undo_split);
            continue;
        }

        apta_node *old_child = old_guard->target->find();

        undo_split(new_child, old_child);

        old_child->data->split_undo(new_child->data);
        old_child->size = old_child->size + new_child->size;
        for (tail *t = new_child->tails_head; t != nullptr; t = t->next()) {
            t->split_from->undo_split();
        }
        new_guard->target = nullptr;
        if(new_child->tails_head != nullptr) mem_store::delete_tail(new_child->tails_head);
        new_child->tails_head = nullptr;
        mem_store::delete_node(new_child);
    }
}

bool state_merger::split_init(apta_node* red, tail* t, int attr, int depth, bool evaluate, bool perform, bool test){
    int symbol = t->get_symbol();
    double val  = t->get_value(attr);

    apta_guard* old_guard = red->guard(t);
    if(old_guard == nullptr) { return false; }
    apta_guard* new_guard = nullptr;

    if(perform){
        if(red->performed_splits == nullptr)
            red->performed_splits = new split_list();
        red->performed_splits->push_front(std::pair< tail*, int >(t, attr));
        new_guard = mem_store::create_guard(old_guard);
        old_guard->min_attribute_values[attr] = val;
        new_guard->max_attribute_values[attr] = val;
        red->guards.insert(std::pair<int, apta_guard *>(symbol, new_guard));
    }

    apta_node* blue = old_guard->target;
    if(blue == nullptr) { return false; }
    blue = blue->find();

    int split_count = 0;
    for(tail_iterator it = tail_iterator(blue); *it != nullptr; ++it) {
        tail *t2 = *it;
        if (t2->past_tail->get_value(attr) < val) split_count++;
    }
    if(split_count == 0) { return false;}

    if(split_count == blue->size){
        if(!perform) { return false;}
        new_guard->target = blue;
        old_guard->target = nullptr;
        return false;
    }

    apta_node* old_child = old_guard->target->find();
    apta_node* new_child = mem_store::create_node(old_child);
    for (tail_iterator it = tail_iterator(blue); *it != nullptr; ++it) {
        tail *t2 = *it;
        if (t2->past_tail->get_value(attr) < val) {
            tail *new_tail = mem_store::create_tail(t2);
            t2->split(new_tail);
            new_child->add_tail(new_tail);
            new_child->data->add_tail(new_tail);
            new_child->size++;
        }
    }

    if(new_child->size == 0){
        mem_store::delete_node(new_child);
        return true;
    }

    if(evaluate){
        eval->split_update_score_before(this, blue, new_child);
    }

    blue->size -= new_child->size;
    if(perform){
        blue->data->split_update(new_child->data);
    }

    if(evaluate){
        eval->split_update_score_after(this, blue, new_child);
    }

    if(perform){
        new_guard->target = new_child;
        new_child->source = red;
    }

    split(new_child, blue, depth + 1, evaluate, perform, test);

    if(!perform){
        blue->size += new_child->size;
        for (tail *t3 = new_child->tails_head; t3 != nullptr; t3 = t3->next()) {
            t3->split_from->undo_split();
        }
        mem_store::delete_tail(new_child->tails_head);
        new_child->tails_head = nullptr;
        mem_store::delete_node(new_child);
    }
    return true;
}

void state_merger::undo_split_init(apta_node* red, tail* t, int attr){
    int symbol = t->get_symbol();
    double val  = t->get_value(attr);

    apta_guard* old_guard = red->guard(t);
    auto new_it = red->guards.upper_bound(t->get_symbol());
    if(new_it == red->guards.begin()) return;
    new_it--;
    apta_guard* new_guard = (*new_it).second;
    int sym2 = new_it->first;
    if(symbol != sym2) { std::cerr << "error" << std::endl; assert(false); }
    if(old_guard->min_attribute_values[attr] != val || new_guard->max_attribute_values[attr] != val) { std::cerr << "error" << std::endl; assert(false); }

    apta_node* blue = old_guard->target;
    apta_node *new_child = new_guard->target;

    if(new_guard->target == nullptr){
        // do nothing
    } else if(old_guard->target == nullptr){
        old_guard->target = new_guard->target;
        new_guard->target = nullptr;
    } else {
        blue = blue->find();
        undo_split(new_child, blue);
        blue->size += new_child->size;
        blue->data->split_undo(new_child->data);
        for (tail *t2 = new_child->tails_head; t2 != nullptr; t2 = t2->next()) {
            t2->split_from->undo_split();
        }
        new_guard->target = nullptr;
        mem_store::delete_tail(new_child->tails_head);
        new_child->tails_head = nullptr;
        mem_store::delete_node(new_child);
    }

    if (new_guard->min_attribute_values.find(attr) != new_guard->min_attribute_values.end())
        old_guard->min_attribute_values[attr] = new_guard->min_attribute_values[attr];
    else
        old_guard->min_attribute_values.erase(attr);

    red->guards.erase(new_it);

    mem_store::delete_guard(new_guard);

    red->performed_splits->erase(red->performed_splits->begin());
}

refinement* state_merger::test_split(apta_node* red, tail* t, int attr){
    apta_node* left = red->child(t);
    bool split_result = split_init(red, t, attr, 0, true, false, false);
    double score_result = -1;
    if(split_result){
        apta_node* right = red->child(t);
        split_result = eval->split_compute_consistency(this, left, right);
        score_result = eval->split_compute_score(this, left, right);
    }
    undo_split_init(red, t, attr);
    if(split_result) return mem_store::create_split_refinement(this, score_result, red, t, attr);
    return nullptr;
}

void state_merger::perform_split(apta_node* red, tail* t, int attr){
    num_merges++;
    split_init(red, t, attr, 0, false, true, false);
}

void state_merger::undo_perform_split(apta_node* red, tail* t, int attr){
    undo_split_init(red, t, attr);
    num_merges--;
}

/* END basic state merging routines */

/* BEGIN merge functions called by state merging algorithms */

/* make a given blue state red, and its children blue */
void state_merger::extend(apta_node* blue){
    blue->red = true;
    blue->sink = blue->sink_type();
    if(blue->source->find()->sink != -1) blue->sink = blue->source->find()->sink;
}

/* undo_merge making a given blue state red */
void state_merger::undo_extend(apta_node* blue){
    blue->red = false;
    blue->sink = -1;
}

/* perform a merge, assumed to succeed, no testing for consistency or score computation */
void state_merger::perform_merge(apta_node* left, apta_node* right){
    num_merges++;
    merge_force(left, right);
}

/* undo_merge a merge, assumed to succeed, no testing for consistency or score computation */
void state_merger::undo_perform_merge(apta_node* left, apta_node* right){
    undo_merge(left, right);
    num_merges--;
}

void state_merger::depth_check_init(){
    if(left_depth_map == nullptr){
        left_depth_map = new std::map<int, apta_node*>();
    }
    if(right_depth_map == nullptr){
        right_depth_map = new std::map<int, apta_node*>();
    }

    while(!left_depth_map->empty()){
        apta_node* n = left_depth_map->begin()->second;
        mem_store::delete_node(n);
        left_depth_map->erase(left_depth_map->begin());
    }

    while(!right_depth_map->empty()){
        apta_node* n = right_depth_map->begin()->second;
        mem_store::delete_node(n);
        right_depth_map->erase(right_depth_map->begin());
    }
}

void state_merger::depth_check_fill(apta_node* node, std::map<int,apta_node*>* depth_map, int depth, bool use_symbol){
    if(DEPTH_CHECK_MAX_DEPTH != -1 && depth > DEPTH_CHECK_MAX_DEPTH) return;

    int index = depth;
    if(use_symbol) index = node->get_access_trace()->get_end()->get_symbol();

    if(depth_map->find(index) == depth_map->end()){
        depth_map->insert(std::pair<int, apta_node*>(index, mem_store::create_node(node)));
    }
    apta_node* depth_node = depth_map->find(index)->second;
    depth_node->data->update(node->data);

    for(auto & it : node->guards){
        if (it.second->target == nullptr) continue;
        apta_node* next = it.second->target;
        depth_check_fill(next, depth_map, depth + 1, use_symbol);
    }
}

bool state_merger::depth_check_run(apta_node* left, apta_node* right, bool use_symbol){
    depth_check_init();
    depth_check_fill(right, right_depth_map, 0, use_symbol);
    bool reset_max_depth = false;
    if(DEPTH_CHECK_MAX_DEPTH == -1){
        reset_max_depth = true;
        DEPTH_CHECK_MAX_DEPTH = right_depth_map->size();
    }
    depth_check_fill(left, left_depth_map, 0, use_symbol);

    eval->reset(this);
    for(int depth = 0; depth <= DEPTH_CHECK_MAX_DEPTH; depth++){
        if(left_depth_map->find(depth) == left_depth_map->end()) continue;
        if(right_depth_map->find(depth) == right_depth_map->end()) continue;

        apta_node* left_depth_node = left_depth_map->find(depth)->second;
        apta_node* right_depth_node = right_depth_map->find(depth)->second;

        bool merge_result = merge_test(left_depth_node, right_depth_node);
        if(!merge_result){
            if(reset_max_depth) DEPTH_CHECK_MAX_DEPTH = -1;
            return false;
        }
    }
    if(reset_max_depth) DEPTH_CHECK_MAX_DEPTH = -1;
    if(!eval->compute_consistency(this, left, right)) return false;
    return true;
}

bool state_merger::pre_consistent(apta_node* left, apta_node* right){
    if(!eval->pre_consistent(this, left, right)) return false;
    if(!MERGE_ROOT && left->source == nullptr) return false;
    if(!MERGE_SINKS && (left->is_sink() || right->is_sink())) return false;
    if(!MERGE_SINKS_WITH_CORE && ((left->is_red() && !left->is_sink()) && right->is_sink())) return false;
    if(MARKOVIAN_MODEL > -1 && !is_path_identical(left, right, MARKOVIAN_MODEL)) return false;
    if(IDENTICAL_KTAIL > -1 && !is_tree_identical(left, right, IDENTICAL_KTAIL)) return false;
    if(left->is_sink()) {
        bool sink_check = false;
        if (CONVERT_SINK_STATES && eval->sink_convert_consistency(this, left, right)) sink_check = true;
        if (MERGE_IDENTICAL_SINKS && is_tree_identical(left, right,-1)) sink_check = true;
        if ((CONVERT_SINK_STATES || MERGE_IDENTICAL_SINKS) && !sink_check) return false;
    }
    if(MERGE_LOCAL > 0 && merged_apta_distance(left, right,-1) > MERGE_LOCAL){
        if(MERGE_LOCAL_COLLECTOR_COUNT == -1 || num_distinct_sources(left) < MERGE_LOCAL_COLLECTOR_COUNT) return false;
    }
    if(STAR_FREE && counting_path_occurs(left, right)){
        return false;
    }
    return true;
}

/* test a merge, behavior depending on input parameters
 * it performs a merge, computes its consistency and score, and undos the merge
 * returns a <consistency,score> pair */
refinement* state_merger::test_merge(apta_node* left, apta_node* right){
    eval->reset(this);

    if(!pre_consistent(left, right)) return nullptr;

    double score_result = -1;
    bool   merge_result = true;

    if(eval->compute_before_merge) score_result = eval->compute_score(this, left, right);
    if(PERFORM_MERGE_CHECK){
        if(MERGE_WHEN_TESTING) merge_result = merge(left,right);
        else merge_result = merge_test(left,right);
    }
    if(merge_result && !eval->compute_before_merge) score_result = eval->compute_score(this, left, right);
    if(USE_LOWER_BOUND && score_result < LOWER_BOUND) merge_result = false;
    if((merge_result && !eval->compute_consistency(this, left, right))) merge_result = false;
    if(PERFORM_MERGE_CHECK && MERGE_WHEN_TESTING) undo_merge(left,right);

    if(PERFORM_DEPTH_CHECK && merge_result){
        merge_result = depth_check_run(left, right, false);
    }
    if(PERFORM_SYMBOL_CHECK && merge_result){
        merge_result = depth_check_run(left, right, true);
    }

    if(!merge_result) return nullptr;
    return mem_store::create_merge_refinement(this, score_result, left, right);
}

refinement* state_merger::test_splits(apta_node* blue){
    refinement* result = nullptr;

    double score = 0;
    double best_score = -1.0;
    for(int attr = 0; attr < dat->get_num_attributes(); ++attr){
        if(!dat->is_splittable(attr)) continue;

        eval->reset_split(this, blue);
        std::multimap<float, tail*> sorted_tails;
        for(tail_iterator it = tail_iterator(blue); *it != nullptr; ++it){
            tail* t = *it;
            sorted_tails.insert(std::pair<float, tail*>(t->past_tail->get_value(attr),t));
        }
        
        apta_node* new_node = mem_store::create_node(blue);

        float prev_val = (*(sorted_tails.begin())).first;
        for(auto & sorted_tail : sorted_tails){
            if(sorted_tail.first > prev_val){
                score = eval->split_compute_score(this, blue, new_node);
                if(eval->split_compute_consistency(this, blue, new_node) && (score > best_score || result == nullptr)){
                    if(result != nullptr) result->erase();
                    result = mem_store::create_split_refinement(this, score, blue->source->find(), sorted_tail.second->past_tail, attr);
                    best_score = score;
                }
                prev_val = sorted_tail.first;
            }
            tail* t = sorted_tail.second;

            eval->split_update_score_before(this, blue, new_node, t);

            blue->size--;
            blue->data->del_tail(t);
            tail* new_tail = mem_store::create_tail(t);
            t->split(new_tail);
            new_node->size++;
            new_node->add_tail(new_tail);
            new_node->data->add_tail(new_tail);
            split_single(new_node, blue, new_tail, 0, true, true, true);

            eval->split_update_score_after(this, blue, new_node, t);

        }

        undo_split_single(new_node, blue);

        while(!temporary_node_store.empty()){
            apta_node* new_child = temporary_node_store.front();
            temporary_node_store.pop_front();
            mem_store::delete_tail(new_child->tails_head);
            new_child->tails_head = nullptr;
            mem_store::delete_node(new_child);
        }

        mem_store::delete_tail(new_node->tails_head);
        new_node->tails_head = nullptr;
        mem_store::delete_node(new_node);
        sorted_tails.clear();
    }

    return result;
}

/**
 * @brief Returns all possible refinements.
 *
 * Returns all refinements given the current sets of red and blue states
 * behavior depends on input parameters
 * the merge score is used as key in the returned refinement_set
 * returns an empty set if none exists (given the input parameters)
 */

refinement_set* state_merger::get_possible_refinements(){
    auto* result = new refinement_set();
    
    state_set blue_its = state_set();
    bool found_non_sink = false;
    
    for(blue_state_iterator it = blue_state_iterator(aut->root); *it != nullptr; ++it){
        if((*it)->size != 0) blue_its.insert(*it);
        if(sink_type(*it) == -1) found_non_sink = true;
    }

    if(!found_non_sink && !MERGE_SINKS){
        return result;
    }

    if(DFA_SIZE_BOUND != -1 && get_num_red_states() >= DFA_SIZE_BOUND){
        return result;
    }
    if(APTA_SIZE_BOUND != -1 && get_final_apta_size() <= APTA_SIZE_BOUND){
        return result;
    }

    for(auto it = blue_its.begin(); it != blue_its.end(); ++it){
        apta_node* blue = *it;
        bool found = false;

        if(found_non_sink && blue->is_sink()) continue;

        if(!blue->is_sink()){
            if(dat->get_num_attributes() > 0){
                refinement* ref = test_splits(blue);
                if(ref != nullptr){
                    result->insert(ref);
                    found = true;
                }
            }
        }

        for(red_state_iterator it2 = red_state_iterator(aut->root); *it2 != nullptr; ++it2){
            apta_node* red = *it2;

            refinement* ref = test_merge(red,blue);

            if(ref != nullptr){
                result->insert(ref);
                found = true;
            }
        }
        
        if(MERGE_BLUE_BLUE){
            for(auto blue2 : blue_its){
                if(blue == blue2) continue;

                if(blue2->is_sink()) continue;

                refinement* ref = test_merge(blue2,blue);
                if(ref != nullptr){
                    result->insert(ref);
                    found = true;
                }
            }
        }
        
        if(EXTEND_ANY_RED && !found && !blue->is_sink()) {
            for(auto it3 : *result) it3->erase();
            result->clear();
            result->insert(mem_store::create_extend_refinement(this, blue));
            return result;
        }
        
        if(!found || EXTEND_SINKS || !blue->is_sink()){
            result->insert(mem_store::create_extend_refinement(this, blue));
        }

        if(MERGE_MOST_VISITED) break;
    }
    return result;
}

/**
 * @brief Returns the highest scoring refinement.
 * 
 * Returns the highest scoring refinement given the current sets of red and blue states
 * behavior depends on input parameters
 * returns 0 if none exists (given the input parameters)
 */
refinement* state_merger::get_best_refinement() {
    refinement_set *rs = get_possible_refinements();
    refinement *r = nullptr;
    if (!rs->empty()) {
        r = *rs->begin();
        for(auto it : *rs){
            if(r != it) it->erase();
        }
    }

    delete rs;
    return r;
}

double state_merger::get_best_refinement_score() {
    refinement *r = get_best_refinement();
    double result = r->score;
    r->erase();
    return result;
}

/* END merge functions called by state merging algorithms */

/* input function 	        *
 * pass along to  eval fct      */

// batch mode methods

/* output functions */
void state_merger::todot(){
    std::stringstream dot_output_buf;
    aut->print_dot(dot_output_buf);
    //dot_output = "// produced with flexfringe from git commit"  + string(gitversion) + '\n' + "// " + COMMAND + '\n'+ dot_output_buf.str();
    dot_output = "// produced with flexfringe // " + COMMAND + '\n'+ dot_output_buf.str();
}

void state_merger::tojson(){
    std::stringstream output_buf;
    aut->print_json(output_buf);
    json_output = output_buf.str(); // json does not support comments, maybe we need to introduce a meta section
}

void state_merger::tojsonsinks(){
    std::stringstream output_buf;
    aut->print_sinks_json(output_buf);
    json_output = output_buf.str(); // json does not support comments, maybe we need to introduce a meta section
}

void state_merger::print_json(FILE* output)
{
    fprintf(output, "%s", json_output.c_str());
}

void state_merger::print_dot(FILE* output)
{
    fprintf(output, "%s", dot_output.c_str());
}

void state_merger::print_dot(const std::string& file_name)
{
    todot();
    std::ofstream output1(file_name.c_str());
    if (output1.fail()) {
        throw std::ofstream::failure("Unable to open file for writing: " + file_name);
    }
    output1 << dot_output;
    output1.close();
}

void state_merger::print_dot(std::ostream& output)
{
    todot();
    output << dot_output;
}

void state_merger::print_json(const std::string& file_name)
{
    tojson();
    std::ofstream output1(file_name.c_str());
    if (output1.fail()) {
        throw std::ofstream::failure("Unable to open file for writing: " + file_name);
    }
    output1 << json_output;
    output1.close();
}

int state_merger::sink_type(apta_node* node){
    if(USE_SINKS)
        return node->data->sink_type();
    return -1;
}

bool state_merger::sink_consistent(apta_node* node, int type){
    return node->data->sink_consistent(type);
}

int state_merger::num_sink_types(){
    return aut->root->data->num_sink_types();
} // this got moved to eval data */

state_merger::~state_merger(){
//    delete aut;
//    delete eval;
//    delete dat;
    std::cerr << "deleted merger" << std::endl;
}

int state_merger::get_num_merges() {
    return num_merges;
}
