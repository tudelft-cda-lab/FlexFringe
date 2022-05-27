#include "apta.h"
#include <iostream>
#include <set>
#include <list>
#include <map>
#include <unordered_map>
#include <string>
#include <iterator>

#include "parameters.h"
#include "evaluation_factory.h"

#include "utility/loguru.hpp"

using namespace std;

inline void apta_node::set_child(tail* t, apta_node* node){
    set_child(t->get_symbol(), node);
};

bool apta_node::is_tree_identical(apta_node* other, int max_depth){
    if(max_depth == 0) return true;
    if(other == nullptr) return false;
    other = other->find();

    for(auto & guard : other->guards) {
        if(guard.second->target != nullptr){
            if(child(guard.first) == nullptr) return false;
            if(!child(guard.first)->find()->is_tree_identical(guard.second->target->find(), max_depth - 1)) return false;
        }
    }
    for(auto & guard : guards) {
        if(guard.second->target != nullptr){
            if(other->child(guard.first) == nullptr) return false;
        }
    }
    return true;
}

bool apta_node::is_path_identical(apta_node* other, int max_depth){
    int ngram = max_depth;
    apta_node* l = this;
    apta_node* r = other;
    while(ngram != 0 && l != nullptr && r != nullptr){
        l = l->find();
        r = r->find();

        if(l->access_trace->get_end()->get_symbol() != r->access_trace->get_end()->get_symbol()) return false;

        ngram--;
        l = l->source;
        r = r->source;
        if(l == nullptr && r == nullptr) return true;
        if(l == nullptr || r == nullptr) return false;
    }
    return true;
}

bool apta_guard::bounds_satisfy(tail* t){
    for(auto & min_attribute_value : min_attribute_values){
        if(t->get_value(min_attribute_value.first) < min_attribute_value.second) return false;
    }
    for(auto & max_attribute_value : max_attribute_values){
        if(t->get_value(max_attribute_value.first) >= max_attribute_value.second) return false;
    }
    return true;
}


/* constructors and destructors */
apta::apta(){

    LOG_S(INFO) << "Creating APTA data structure";
    root = new apta_node();
    root->red = true;
    merger = nullptr;
}

void apta::print_dot(iostream& output){
    output << "digraph DFA {\n";
    output << "\t" << root->find()->number << " [label=\"root\" shape=box];\n";
    output << "\t\tI -> " << root->find()->number << ";\n";
    int ncounter = 0;
    for(APTA_iterator Ait = APTA_iterator(root); *Ait != 0; ++Ait){
        apta_node* n = *Ait;
        n->number = ncounter++;
    }
    for(APTA_iterator Ait = APTA_iterator(root); *Ait != 0; ++Ait){
        apta_node *n = *Ait;
        if(!DEBUGGING && n->rep() != nullptr) continue;

        if (!n->data->print_state_true()) {
            continue;
        }

        if (!PRINT_WHITE && !n->red) {
            if (n->source != nullptr) {
                if (!n->source->find()->red)
                    continue;
                if (!PRINT_BLUE)
                    continue;
            }
        }

        output << "\t" << n->number << " [ label=\"";
        if(DEBUGGING){
            output << n << ":#" << "\n";
            output << "rep#" << n->representative << "\n";
        }
        output << n->number << ":#" << n->size << "\n";
        n->data->print_state_label(output);
        output << "\" ";
        n->data->print_state_style(output);
        if (!n->red) output << " style=dotted";
        output << ", penwidth=" << log(1 + n->size);
        output << "];\n";

        for(auto it = n->guards.begin(); it != n->guards.end(); ++it){
            if(it->second->target == nullptr) continue;

            apta_guard* g = it->second;
            apta_node* child = it->second->target->find();
            if(DEBUGGING) child = it->second->target;

            if (!PRINT_WHITE && !child->red) {
                if (!n->red)
                    continue;
                if (!PRINT_BLUE)
                    continue;
            }

            output << "\t\t" << n->number << " -> " << child->number << " [label=\"";

            output << inputdata::get_symbol(it->first) << endl;

            n->data->print_transition_label(output, it->first);
            
            
            for(auto & min_attribute_value : g->min_attribute_values){
                output << " " << inputdata::get_attribute(min_attribute_value.first) << " >= " << min_attribute_value.second;
            }
            for(auto & max_attribute_value : g->max_attribute_values){
                output << " " << inputdata::get_attribute(max_attribute_value.first) << " < " << max_attribute_value.second;
            }

            output << "\" ";
            output << " ];\n";
        }
    }
    output << "}\n";
}

void apta_node::print_json(iostream& output){
    output << "\t\t{\n";
    output << "\t\t\t\"id\" : " << number << ",\n";
    //output << "\t\t\t\"access\" : " << get_trace_from_state()->to_string() << ",\n";
    if(source != nullptr) output << "\t\t\t\"source\" :  " << source->find()->number << ",\n";
    else output << "\t\t\t\"source\" :  " << -1 << ",\n";
    output << "\t\t\t\"label\" : \"";
    data->print_state_label_json(output);
    output  << "\",\n";
    output << "\t\t\t\"size\" : " << size << ",\n";
    output<< "\t\t\t\"level\" : " << depth << ",\n";
    output << "\t\t\t\"style\" : \"";
    data->print_state_style(output);
    output  << "\",\n";
    output << "\t\t\t\"isred\" :  " << is_red() << ",\n";
    output << "\t\t\t\"issink\" :  " << is_sink() << ",\n";
    output << "\t\t\t\"isblue\" :  " << is_blue() << ",\n";
    output << "\t\t\t\"trace\" :  \"" << access_trace->to_string() << "\",\n";
    json d;
    data->write_json(d);
    output << "\t\t\t\"data\" :  " << d;
    output << "\n\t\t}";
}

void apta_node::print_json_transitions(iostream& output){
    bool first = true;
    for(auto & guard : guards){
        if(guard.second->target == nullptr) continue;
        apta_node* child = guard.second->target->find();

        if(!first) output << ",\n";
        else first = false;


        output << "\t\t{\n";
        output << "\t\t\t\"id\" : \"" << number << "_" << child->number << "\",\n";
        output << "\t\t\t\"source\" : \"" << number << "\",\n";
        output << "\t\t\t\"target\" : \"" << child->number << "\",\n";

        output << "\t\t\t\"name\": \"" << inputdata::get_symbol(guard.first) << "\",\n";
        output << "\t\t\t\"appearances\": \"";
        data->print_transition_label_json(output, guard.first);
        output << "\"}\n";
    }
}

void apta::print_json(iostream& output){
    set_json_depths();
    int count = 0;
    root->depth = 0;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait){
        apta_node* n = *Ait;
        n->number = count++;
    }

    output << "{\n";
    output << "\t\"types\" : [\n";
    for (int i = 0; i < inputdata::get_types_size(); ++i) {
        if(i != 0) output << ",\n";
        output << "\"" << inputdata::string_from_type(i) << "\"";
    }
    output << "\n\t],\n";
    output << "\t\"alphabet\" : [\n";
    for (int i = 0; i < inputdata::get_alphabet_size(); ++i) {
        if(i != 0) output << ",\n";
        output << "\"" << inputdata::string_from_symbol(i)<< "\"";
    }
    output << "\n\t],\n";
    output << "\t\"nodes\" : [\n";
    bool first = true;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
        apta_node *n = *Ait;

        if (!PRINT_WHITE && !n->red) {
            if (n->source != nullptr) {
                if (!n->source->find()->red)
                    continue;
                if (!PRINT_BLUE)
                    continue;
            }
        }

        if(!first)
            output << ",\n";
        else
            first = false;
        n->print_json(output);
    }

    output << "\n\t],\n";

    output << "\t\"edges\" : [\n";
    first = true;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
        apta_node *n = *Ait;

        if (!PRINT_WHITE && !n->red) {
            if (n->source != nullptr) {
                if (!n->source->find()->red)
                    continue;
                if (!PRINT_BLUE)
                    continue;
            }
        }

        bool found = false;
        for(auto & guard : n->guards){
            if(guard.second->target != nullptr){
                apta_node* target = guard.second->target->find();
                if (!PRINT_WHITE && !target->red) {
                    if (target->source != nullptr) {
                        if (!target->source->find()->red)
                            continue;
                        if (!PRINT_BLUE)
                            continue;
                    }
                }
                found = true;
                break;
            }
        }
        if(!found) continue;

        if(!first)
            output << ",\n";
        else
            first = false;
        n->print_json_transitions(output);
    }
    output << "\n\t]\n}\n";
}

void apta::print_sinks_json(iostream& output) const{
    output << "{\n";
    output << "\t\"nodes\" : [\n";
    bool first = true;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
        apta_node *n = *Ait;
        if(n->red) continue;

        if (!first) output << ",\n";
        else first = false;

        n->print_json(output);
    }
    output << "\n\t],\n";

    output << "\t\"edges\" : [\n";
    first = true;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
        apta_node *n = *Ait;
        if(n->red) continue;

        bool found = false;
        for(auto & guard : n->guards){
            if(guard.second->target != nullptr){
                found = true;
                break;
            }
        }
        if(!found) continue;

        if (!first) output << ",\n";
        else first = false;

        n->print_json_transitions(output);
    }
    output << "\n\t]\n}\n";
}

void apta::read_json(istream& input_stream){
    json read_apta = json::parse(input_stream);
    inputdata idat;

    map<int, apta_node*> states;
    //for each json line
    for (auto & i : read_apta["types"]) {
        inputdata::type_from_string(i);
    }
    for (auto & i : read_apta["alphabet"]) {
        inputdata::symbol_from_string(i);
    }
    for (int i = 0; i < read_apta["nodes"].size(); ++i) {
        json n = read_apta["nodes"][i];
        auto *node = new apta_node();
        states[n["id"]] = node;
        int r = n["isred"];
        node->red = r;
        if (n["id"] == 0) {
            root = node;
        }
        node->number = n["id"];
        node->size = n["size"];
        node->data->read_json(n["data"]);
        node->source = states[n["source"]];
        string trace = n["trace"];
        istringstream trace_stream(trace);
        node->access_trace = mem_store::create_trace();
        idat.read_abbadingo_sequence(trace_stream,node->access_trace);
    }
    for (int i = 1; i < read_apta["nodes"].size(); ++i) {
        json n = read_apta["nodes"][i];
        apta_node *node = states[n["id"]];
        if(n["source"] != -1)
            node->source = states[n["source"]];
        else
            node->source = nullptr;
    }
    for (int i = 0; i < read_apta["edges"].size(); ++i) {
        json e = read_apta["edges"][i];

        string symbol = e["name"];
        //if symbol not in alphabet, add it
        int symbol_nr = inputdata::symbol_from_string(symbol);

        string source_string = e["source"];
        string target_string = e["target"];

        int source_nr = std::stoi(source_string);
        int target_nr = std::stoi(target_string);

        if(states.find(source_nr) == states.end()) continue;
        if(states.find(target_nr) == states.end()) continue;

        apta_node* source = states[source_nr];
        apta_node* target = states[target_nr];

        if(target->source == source) {
            source->set_child(symbol_nr, target);
        } else {
            auto* new_target = new apta_node();
            new_target->source = source;
            new_target->red = false;
            new_target->size = 0;
            source->set_child(symbol_nr, new_target);
            new_target->merge_with(target);
        }
    }
}

apta_guard::apta_guard(){
    undo_split = nullptr;
    target = nullptr;
}

apta_guard::apta_guard(apta_guard* g){
    undo_split = nullptr;
    target = nullptr;

    min_attribute_values = bound_map(g->min_attribute_values);
    max_attribute_values = bound_map(g->max_attribute_values);
}

void apta_guard::initialize(apta_guard* g){
    undo_split = nullptr;
    target = nullptr;

    min_attribute_values.clear();
    max_attribute_values.clear();

    min_attribute_values.insert(g->min_attribute_values.begin(), g->min_attribute_values.end());
    max_attribute_values.insert(g->max_attribute_values.begin(), g->max_attribute_values.end());
}

void apta_node::add_tail(tail* t){
    t->next_in_list = tails_head;
    tails_head = t;
}

apta_node::apta_node(){
    source = nullptr;
    original_source = nullptr;
    representative = nullptr;
    next_merged_node = nullptr;
    representative_of = nullptr;

    performed_splits = nullptr;
    tails_head = nullptr;
    access_trace = nullptr;
    number = -1;
    size = 0;
    final = 0;
    depth = 0;
    red = false;
    sink = -1;

    try {
       data = (DerivedDataRegister<evaluation_data>::getMap())->at(DATA_NAME)();
       data->node = this;
    } catch(const std::out_of_range& oor ) {
       std::cerr << "No data type found..." << std::endl;
    }
}

void apta_node::initialize(apta_node* n){
    source = nullptr;
    original_source = nullptr;
    representative = nullptr;
    next_merged_node = nullptr;
    representative_of = nullptr;
    tails_head = nullptr;
    access_trace = nullptr;
    number = -1;
    size = 0;
    final = 0;
    depth = 0;
    red = false;
    sink = -1;
    data->initialize();
    for(auto & guard : guards){
        mem_store::delete_guard(guard.second);
    }
    guards.clear();
    if(performed_splits != nullptr) performed_splits->clear();
}


apta_node* apta_node::child(tail* t){
        int symbol = t->get_symbol();
        for(auto it = guards.lower_bound(symbol); it != guards.upper_bound(symbol); ++it){
            if(it->first != symbol) break;
            apta_guard* g = it->second;
            bool outside_range = false;
            for(auto & min_attribute_value : g->min_attribute_values){
                if(t->get_value(min_attribute_value.first) < min_attribute_value.second){
                    outside_range = true;
                    break;
                }
            }
            if(outside_range) continue;
            for(auto & max_attribute_value : g->max_attribute_values){
                if(t->get_value(max_attribute_value.first) >= max_attribute_value.second){
                    outside_range = true;
                    break;
                }
            }
            if(outside_range) continue;
            return g->target;
        }
        return nullptr;
}

apta_guard* apta_node::guard(int symbol, apta_guard* g){
        for(auto it = guards.lower_bound(symbol); it != guards.upper_bound(symbol); ++it){
            if(it->first != symbol) break;
            apta_guard* g2 = it->second;
            bool outside_range = false;
            for(auto & min_attribute_value : g->min_attribute_values){
                auto it3 = g2->min_attribute_values.find(min_attribute_value.first);
                if(it3 == g2->min_attribute_values.end() || (*it3).second != min_attribute_value.second){
                    outside_range = true;
                    break;
                }
            }
            if(outside_range) continue;
            for(auto & max_attribute_value : g->max_attribute_values){
                auto it3 = g2->max_attribute_values.find(max_attribute_value.first);
                if(it3 == g2->max_attribute_values.end() || (*it3).second != max_attribute_value.second){
                    outside_range = true;
                    break;
                }
            }
            if(outside_range) continue;
            return g2;
        }
        return nullptr;
}

apta_guard* apta_node::guard(tail* t){
    int symbol = t->get_symbol();
    auto it = guards.lower_bound(symbol);
    auto it_end = guards.upper_bound(symbol);
    for(;it != it_end; ++it){
        if(it->second->bounds_satisfy(t)){
            return it->second;
        }
    }
    return nullptr;
}

int apta_node::apta_distance(apta_node* right, int bound){
    int dist = 0;
    apta_node* l = this;
    apta_node* r = right;
    while(l != nullptr && r != nullptr){
        if(l == r) break;

        if(r->depth > l->depth) r = r->source;
        else l = l->source;
        dist++;
        if(bound != -1 && dist >= bound) return dist;
    }
    return dist;
}

int apta_node::merged_apta_distance(apta_node* right, int bound){
    int dist = 0;
    apta_node* l = this;
    apta_node* r = right;
    while(l != nullptr && r != nullptr){
        l = l->find();
        r = r->find();
        if(l == r) break;

        if(r->depth > l->depth) r = r->source;
        else l = l->source;
        dist++;
        if(bound != -1 && dist >= bound) return bound;
    }
    return dist;
}

int apta_node::num_distinct_sources(){
    set<apta_node*> sources;
    sources.insert(find());
    if(source != nullptr){
        sources.insert(source->find());
        for(apta_node* n = representative_of; n != nullptr; n = n->next_merged_node){
            if(n->source != nullptr){
                sources.insert(n->source->find());
            }
        }
    }
    return (int) sources.size();
}

/* iterators for the APTA and merged APTA */

APTA_iterator::APTA_iterator(apta_node* start){
    base = start;
    current = start;
}

void APTA_iterator::increment() {
    guard_map::iterator it;
    for(it = current->guards.begin();it != current->guards.end(); ++it) {
        apta_node *target = it->second->target;
        if (target != nullptr && target->source == current) {
            q.push(target);
        }
    }
    if(!q.empty()) {
        current = q.front();
        q.pop();
    } else {
        current = nullptr;
    }
}

merged_APTA_iterator::merged_APTA_iterator(apta_node* start){
    base = start;
    current = start;
}

void merged_APTA_iterator::increment() {
    guard_map::iterator it;
    for(it = current->guards.begin();it != current->guards.end(); ++it) {
        apta_node *target = it->second->target;
        if (target != nullptr && target->source->find() == current && target->representative == nullptr) {
            q.push(target);
        }
    }
    if(!q.empty()) {
        current = q.front();
        q.pop();
    } else {
        current = nullptr;
    }
}

merged_APTA_iterator_func::merged_APTA_iterator_func(apta_node* start, bool(*node_check)(apta_node*)) : merged_APTA_iterator(start){
    check_function = node_check;
}

void merged_APTA_iterator_func::increment() {
    guard_map::iterator it;
    for(it = current->guards.begin();it != current->guards.end(); ++it) {
        apta_node *target = it->second->target;
        if (target != nullptr && target->source->find() == current && target->representative == nullptr) {
            if(!check_function(current)) q.push(target);
        }
    }
    if(!q.empty()) {
        current = q.front();
        q.pop();
    } else {
        current = nullptr;
    }
}

blue_state_iterator::blue_state_iterator(apta_node* start) : merged_APTA_iterator(start) {
    if(current->red) blue_state_iterator::increment();
}

void blue_state_iterator::increment() {
    if(current->red) {
        guard_map::iterator it;
        for (it = current->guards.begin(); it != current->guards.end(); ++it) {
            apta_node *target = it->second->target;
            if (target != nullptr && target->source->find() == current && target->representative == nullptr) {
                q.push(target);
            }
        }
    }
    if(!q.empty()) {
        current = q.front();
        q.pop();
        if(current->red) increment();
    } else {
        current = nullptr;
    }
}

red_state_iterator::red_state_iterator(apta_node* start) : merged_APTA_iterator(start) { }

void red_state_iterator::increment() {
    if(current->red) {
        guard_map::iterator it;
        for (it = current->guards.begin(); it != current->guards.end(); ++it) {
            apta_node *target = it->second->target;
            if (target != nullptr && target->source->find() == current && target->representative == nullptr) {
                if(target->red) q.push(target);
            }
        }
    }
    if(!q.empty()) {
        current = q.front();
        q.pop();
    } else {
        current = nullptr;
    }
}

tail_iterator::tail_iterator(apta_node* start){
    base = start;
    current = start;
    current_tail = current->tails_head;
    while(current_tail == nullptr){
        if(current == nullptr) return;
        next_node();
        if(current == nullptr) return;
        current_tail = current->tails_head;
    }
    if(current_tail != nullptr && current_tail->split_to != nullptr) tail_iterator::increment();
}

void tail_iterator::next_node(){
    if(current->representative_of != nullptr) current = current->representative_of;
    else if (current->next_merged_node != nullptr && base != current) current = current->next_merged_node;
    else {
        while(current != nullptr && current->next_merged_node == nullptr) current = current->representative;
        if(current != nullptr && base != current) current = current->next_merged_node;
        else current = nullptr;
    }
}

void tail_iterator::increment() {
    current_tail = current_tail->next_in_list;
    while(current_tail != nullptr && current_tail->split_to != nullptr) current_tail = current_tail->next_in_list;
    
    while(current_tail == nullptr){
        if(current == nullptr) return;
        next_node();
        if(current == nullptr) return;
        current_tail = current->tails_head;
    }
    if(current_tail != nullptr && current_tail->split_to != nullptr) increment();
}


apta_node* tail_iterator::next_forward(){
    current_tail = current_tail->next_in_list;
    if(current_tail != nullptr && current_tail->split_to != nullptr) increment();

    while(current_tail == nullptr){
        if(current == nullptr) return nullptr;
        next_node();
        if(current == nullptr) return nullptr;
        current_tail = current->tails_head;
    }
    if(current_tail != nullptr && current_tail->split_to != nullptr) increment();

    return current;
}

apta::~apta(){
    state_set states;
    for(APTA_iterator Ait = APTA_iterator(root); *Ait != 0; ++Ait){
        states.insert(*Ait);
    }
    for(auto st : states){
        delete st;
    }
}

apta_node::~apta_node(){
    if(access_trace != nullptr) delete access_trace;
    for(auto & guard : guards){
        delete guard.second;
    }
    /* deleted in input_data
    tail* t = tails_head;
    while(t != nullptr){
        tail* n = t->next_in_list;
        delete t;
        t = n;
    }*/
    delete data;
}

void apta::set_json_depths() const {
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait){
        apta_node* n = *Ait;
        if(n->source == nullptr) n->depth = 0;
        else n->depth = n->source->find()->depth + n->num_distinct_sources();
    }
    set<int> depths;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait){
        apta_node* n = *Ait;
        for(apta_node* n2 = n->representative_of; n2 != nullptr; n2 = n2->next_merged_node){
            if(n2->source != nullptr && n2->source->find() != n->find()){
                if(n2->source->find()->depth == n->depth){
                    n->depth = n->depth + 1;
                }
            }
        }
        depths.insert(n->depth);
    }
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
        apta_node* n = *Ait;
        auto it = depths.begin();
        for(int i = 0; i < depths.size(); ++i){
            if(n->depth == *it) {
                n->depth = i;
                break;
            }
            it++;
        }
    }
}