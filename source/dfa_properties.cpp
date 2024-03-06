//
// Created by sicco on 07/07/2022.
//

#include "dfa_properties.h"
#include "apta.h"

bool is_counting_path(std::string str){
    return ((str + str).find(str, 1) != str.size());
}

bool counting_path_occurs(apta_node* n1, apta_node* n2){
    std::map<int,tail*> tail_map;
    for(auto it = tail_iterator(n2); *it != nullptr; ++it){
        tail_map.insert(std::pair<int, tail*>((*it)->get_sequence(), nullptr));
    }
    for(auto it = tail_iterator(n1); *it != nullptr; ++it){
        auto hit = tail_map.find((*it)->get_sequence());
        if(hit != tail_map.end()) hit->second = *it;
    }
    std::set<std::string> traces;
    for(auto it = tail_map.begin(); it != tail_map.end(); ++it) {
        tail *t = it->second;
        if(t == nullptr) continue;

        std::string sequence;
        apta_node *node = n1;
        while (node != n2 && node != nullptr) {
            sequence.push_back(t->get_symbol());
            node = node->child(t);
            t = t->future();
        }
        if(traces.find(sequence) == traces.end()) traces.insert(sequence);
    }
    for(auto sequence : traces){
        if(is_counting_path(sequence)) return true;
    }
    return false;
}

bool is_tree_identical(apta_node* l, apta_node* r, int max_depth){
    if(max_depth == 0) return true;
    if(l == nullptr || r == nullptr) return false;
    r = r->find();

    for(auto it = r->guards_start(); it != r->guards_end(); ++it) {
        if(it->second->get_target() != nullptr){
            if(l->child(it->first) == nullptr) return false;
            if(!is_tree_identical(l->child(it->first)->find(), it->second->get_target()->find(), max_depth - 1)) return false;
        }
    }
    for(auto it = r->guards_start(); it != r->guards_end(); ++it) {
        if(it->second->get_target() != nullptr){
            if(r->child(it->first) == nullptr) return false;
        }
    }
    return true;
}

bool is_path_identical(apta_node* l, apta_node* r, int max_depth){
    int ngram = max_depth;
    while(ngram != 0 && l != nullptr && r != nullptr){
        l = l->find();
        r = r->find();

        if(l->get_access_trace()->get_end() == nullptr && r->get_access_trace()->get_end() != nullptr) return false;
        if(l->get_access_trace()->get_end() != nullptr && r->get_access_trace()->get_end() == nullptr) return false;
        if(l->get_access_trace()->get_end() == nullptr && r->get_access_trace()->get_end() == nullptr) return true;
        if(l->get_access_trace()->get_end()->get_symbol() != r->get_access_trace()->get_end()->get_symbol()) return false;

        ngram--;
        l = l->get_source();
        r = r->get_source();
        if(l == nullptr && r == nullptr) return true;
        if(l == nullptr || r == nullptr) return false;
    }
    return true;
}

int apta_distance(apta_node* l, apta_node* r, int bound){
    int dist = 0;
    while(l != nullptr && r != nullptr){
        if(l == r) break;

        if(r->get_depth() > l->get_depth()) r = r->get_source();
        else l = l->get_source();
        dist++;
        if(bound != -1 && dist >= bound) return dist;
    }
    return dist;
}

int merged_apta_distance(apta_node* l, apta_node* r, int bound){
    int dist = 0;
    while(l != nullptr && r != nullptr){
        l = l->find();
        r = r->find();
        if(l == r) break;

        if(r->get_depth() > l->get_depth()) r = r->get_source();
        else l = l->get_source();
        dist++;
        if(bound != -1 && dist >= bound) return bound;
    }
    return dist;
}

int num_distinct_sources(apta_node* node){
    std::set<apta_node*>* sources = node->get_sources();
    int result = sources->size();
    delete sources;
    return result;
}

