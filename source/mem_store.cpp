
#include "mem_store.h"
#include "input/inputdatalocator.h"
#include <iostream>

std::list< apta_node* > mem_store::node_store;
std::list< apta_guard* > mem_store::guard_store;
std::list< tail* > mem_store::tail_store;
std::list< trace* > mem_store::trace_store;
std::list< merge_refinement* > mem_store::mergeref_store;
std::list< split_refinement* > mem_store::splitref_store;
std::list< extend_refinement* > mem_store::extendref_store;

void mem_store::delete_node(apta_node* node){
    assert(node != nullptr);
    node_store.push_front(node);
    //cerr << "delete " << node << endl;
    //delete node;
};
apta_node* mem_store::create_node(apta_node* other_node){
    apta_node* node = 0;
    if(!node_store.empty()){
        node = node_store.front();
        node_store.pop_front();
        node->initialize(other_node);
    } else { node = new apta_node(); }
    //cerr << "create " << node << endl;
    return node;
};

void mem_store::delete_guard(apta_guard* guard){
    assert(guard != nullptr);
    guard_store.push_front(guard);
};
apta_guard* mem_store::create_guard(apta_guard* other_guard){
    apta_guard* guard = 0;
    if(!guard_store.empty()){
        guard = guard_store.front();
        guard_store.pop_front();
        guard->initialize(other_guard);
    } else { guard = new apta_guard(other_guard); }
    return guard;
};
//
//void mem_store::delete_tail(tail* t){
//    assert(t != nullptr);
//    tail_store.push_front(t);
//};
//tail* mem_store::create_tail(tail* other_tail){
//    tail* t = 0;
//    if(!tail_store.empty()){
//        tail* t2 = tail_store.front();
//        if(t2->next_in_list == 0) { tail_store.pop_front(); t = t2; }
//        else { t = t2->next_in_list; t2->next_in_list = t->next_in_list; }
//        t->initialize(other_tail);
//    } else { t = new tail(other_tail); }
//    return t;
//};

//void mem_store::delete_trace(trace* tr){
//    assert(tr != nullptr);
//    tail* t = tr->head;
//    while(t != 0){
//        mem_store::delete_tail(t);
//        tail* t2 = t;
//        t = t2->future();
//    }
//    trace_store.push_front(tr);
//};
//trace* mem_store::create_trace(){
//    trace* t;
//    if(!trace_store.empty()){
//        t = trace_store.front();
//        trace_store.pop_front();
//        t->initialize();
//    } else {
//        t = new trace();
//    }
//    return t;
//};

void mem_store::delete_merge_refinement(merge_refinement* ref){
    assert(ref != nullptr);
    mergeref_store.push_front(ref);
};
merge_refinement* mem_store::create_merge_refinement(state_merger* m, double s, apta_node* l, apta_node* r){
    merge_refinement* ref = 0;
    if(!mergeref_store.empty()){
        ref = mergeref_store.front();
        mergeref_store.pop_front();
        ref->initialize(m, s, l, r);
    } else { ref = new merge_refinement(m, s, l, r); }
    return ref;
};

void mem_store::delete_split_refinement(split_refinement* ref){
    assert(ref != nullptr);
    splitref_store.push_front(ref);
};
split_refinement* mem_store::create_split_refinement(state_merger* m, double s, apta_node* l, tail* t, int a){
    split_refinement* ref = 0;
    if(!splitref_store.empty()){
        ref = splitref_store.front();
        splitref_store.pop_front();
        ref->initialize(m, s, l, t, a);
    } else { ref = new split_refinement(m, s, l, t, a); }
    return ref;
};

void mem_store::delete_extend_refinement(extend_refinement* ref){
    assert(ref != nullptr);
    extendref_store.push_front(ref);
};
extend_refinement* mem_store::create_extend_refinement(state_merger* m, apta_node* r){
    extend_refinement* ref = 0;
    if(!extendref_store.empty()){
        ref = extendref_store.front();
        extendref_store.pop_front();
        ref->initialize(m, r);
    } else { ref = new extend_refinement(m, r); }
    return ref;
};

void mem_store::erase(){
    for(auto & it : mem_store::node_store){
        delete it;
    }
    for(auto & it : mem_store::guard_store){
        delete it;
    }
    for(auto t : mem_store::tail_store){
        while(t != 0){
            tail* t2 = t;
            t = t2->next_in_list;
            delete t2;
        }
    }
    for(auto & it : mem_store::mergeref_store){
        delete it;
    }
    for(auto & it : mem_store::splitref_store){
        delete it;
    }
    for(auto & it : mem_store::extendref_store){
        delete it;
    }

}

void mem_store::delete_trace(trace* trace) {
    assert(trace != nullptr);
    tail* t = trace->head;
    while(t != nullptr){
        mem_store::delete_tail(t);
        tail* t2 = t;
        t = t2->future();
    }
    delete[] trace->trace_attr;
    trace_store.push_front(trace);
}

trace *mem_store::create_trace(inputdata* inputData) {
    if (inputData == nullptr) {
        // Will crash if no InputData can be located
        inputData = inputdata_locator::get();
    }

    trace* t;
    if(!trace_store.empty()) {
        t = trace_store.front();
        trace_store.pop_front();
        t->initialize(inputData);
    } else {
        t = new trace(inputData);
    }
    return t;
}

void mem_store::delete_tail(tail * t) {
    assert(t != nullptr);
    tail_store.push_front(t);
}

tail* mem_store::create_tail(tail *other_tail) {
    tail* t = nullptr;
    if(!tail_store.empty()){
        tail* t2 = tail_store.front();
        if(t2->next_in_list == nullptr) { tail_store.pop_front(); t = t2; }
        else { t = t2->next_in_list; t2->next_in_list = t->next_in_list; }
        t->initialize(other_tail);
    } else { t = new tail(other_tail); }
    return t;
};