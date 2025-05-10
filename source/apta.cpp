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
#include "evaluate.h"
#include "utility/loguru.hpp"
#include "input/abbadingoreader.h"
#include "input/inputdatalocator.h"
#include "input/parsers/abbadingoparser.h"

/*inline void apta_node::set_child(tail* t, apta_node* node){
    set_child(t->get_symbol(), node);
};*/

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
    root->number = 0;
}

bool apta::print_node(apta_node* n){
    if(n->rep() != nullptr) return false;
    if(!n->data->print_state_true()) return false;
    if(!PRINT_RED && n->red) return false;
    if(!PRINT_WHITE && !n->red) {
        if (n->source != nullptr) {
            if (!n->source->find()->red) return false;
            if (!PRINT_BLUE) return false;
        }
    }
    return true;
}

void apta_node::print_dot(std::iostream& output){
    output << "\t" << number << " [ label=\"";
    output << number << " #" << size << " ";
    data->print_state_label(output);
    data->print_state_style(output);
    output << "\" ";

    //if (representative == nullptr) output << ", style=filled";
    //else output << ", style=dotted";
    output << ", style=filled";

    if (is_red()) output << ", fillcolor=\"tomato\"";
    else if (is_blue()) output << ", fillcolor=\"lightskyblue\"";
    else if (is_white()) output << ", fillcolor=\"ghostwhite\"";

    output << ", width=" << log(1 + log(1 + size));
    output << ", height=" << log(1 + log(1 + size));
    output << ", penwidth=" << log(1 + size);
    output << "];\n";
}

void apta::print_dot(std::iostream& output){
    output << "digraph DFA {\n";
    output << "\t" << root->find()->number << " [label=\"root\" shape=box];\n";
    output << "\t\tI -> " << root->find()->number << ";\n";
    for(APTA_iterator Ait = APTA_iterator(root); *Ait != nullptr; ++Ait){
        apta_node *n = *Ait;
        if(!print_node(n)) continue;
        n->print_dot(output);

        for(auto it = n->guards.begin(); it != n->guards.end(); ++it){
            if(it->second->target == nullptr) continue;

            apta_guard* g = it->second;
            apta_node* child = it->second->target->find();
            if(!print_node(child)) continue;

            if(DEBUGGING) child = it->second->target;

            output << "\t\t" << n->number << " -> " << child->number << " [label=\"";
            output << inputdata_locator::get()->get_symbol(it->first) << " ";
            n->data->print_transition_label(output, it->first);
            for(auto & min_attribute_value : g->min_attribute_values){
                output << "\\n" << inputdata_locator::get()->get_attribute(min_attribute_value.first) << " >= " << min_attribute_value.second;
            }
            for(auto & max_attribute_value : g->max_attribute_values){
                output << "\\n" << inputdata_locator::get()->get_attribute(max_attribute_value.first) << " < " << max_attribute_value.second;
            }
            output << "\" ";
            //if(child->representative != nullptr) output << ", style=dotted";
            output << ", penwidth=" << log(1 + n->size);
            output << " ];\n";

            if(DEBUGGING){
                child = it->second->target;
                while(child->representative != nullptr){
                    child->print_dot(output);
                    output << "\t\t" << child->number << " -> " << child->representative->number;
                    output << " [label=\"m" << child->merge_score << "\" , penwidth=" << log(1 + n->size) << " ];\n";
                    child = child->representative;
                }
            }
        }
    }
    output << "}\n";
}

void apta_node::print_json(json& nodes){
    json output;
    output["id"] = number;
    if(source != nullptr) output["source"] = source->find()->number;
    else output["source"] = -1;
    output["size"] = size;
    output["level"] = depth;
    output["isred"] = is_red();
    output["isblue"] = is_blue();
    output["issink"] = is_sink();
    output["trace"] = access_trace->to_string();
    if(DEBUGGING){
        if(representative != nullptr){
            output["representative"] = representative->number;
            output["mergescore"] = merge_score;
        } else {
            output["representative"] = -1;
            output["mergescore"] = 0.0;
        }
    }
    json d;
    data->write_json(d);
    output["data"] = d;
    nodes.push_back(output);
}

void apta_node::print_json_transitions(json& edges){
    for(auto & guard : guards){
        json output;
        if(guard.second->target == nullptr) continue;
        apta_node* child = guard.second->target;

        output["source"] = number;
        if(DEBUGGING) output["target"] = child->number;
        else output["target"] = child->find()->number;

        output["label"] = inputdata_locator::get()->get_symbol(guard.first);
        edges.push_back(output);
    }
}

void apta::print_json(std::iostream& outio){
    set_json_depths();
    int count = 0;
    root->depth = 0;

    json output;

    std::list<std::string> types;
    for (int i = 0; i < inputdata_locator::get()->get_types_size(); ++i) {
        types.push_back(inputdata_locator::get()->string_from_type(i));
    }
    output["types"] = types;

    std::list<std::string> alphabet;
    for (int i = 0; i < inputdata_locator::get()->get_alphabet_size(); ++i) {
        alphabet.push_back(inputdata_locator::get()->string_from_symbol(i));
    }
    output["alphabet"] = alphabet;

    json nodes;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
        apta_node *n = *Ait;
        if(!print_node(n)) continue;

        n->print_json(nodes);
        for(auto & guard : n->guards) {
            if (guard.second->target != nullptr){
                apta_node* target = guard.second->target;
                if(!print_node(target->find())) continue;
                if(DEBUGGING){
                    while(target->representative != nullptr){
                        target->print_json(nodes);
                        target = target->representative;
                    }
                } else
                {
                    target = target->find();
                    target->print_json(nodes);
                }
            }
        }
    }
    output["nodes"] = nodes;

    json edges;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
        apta_node *n = *Ait;
        if(!print_node(n)) continue;

        bool found = false;
        for(auto & guard : n->guards){
            if(guard.second->target != nullptr){
                apta_node* target = guard.second->target;
                if(!print_node(target->find())) continue;
                found = true;
                break;
            }
        }
        if(!found) continue;

        n->print_json_transitions(edges);
    }
    output["edges"] = edges;

    outio << output.dump(2);
}

void apta::print_sinks_json(std::iostream& output){
    print_json(output);
}

void apta::read_json(std::istream& input_stream){
    json read_apta = json::parse(input_stream);
    // abbadingo_inputdata idat;

    std::map<int, apta_node*> states;
    //for each json line
    for (auto & i : read_apta["types"]) {
        inputdata_locator::get()->type_from_string(i);
    }
    for (auto & i : read_apta["alphabet"]) {
        inputdata_locator::get()->symbol_from_string(i);
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
        std::string trace = n["trace"];
        std::istringstream trace_stream(trace);
        node->access_trace = mem_store::create_trace();

        auto parser = abbadingoparser::single_trace(trace_stream);
        auto strategy = read_all();
        auto trace_maybe = inputdata_locator::get()->read_trace(parser, strategy);
        if (trace_maybe.has_value()) {
            node->access_trace = trace_maybe.value();
        }
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

        std::string symbol = e["label"];
        //if symbol not in alphabet, add it
        int symbol_nr = inputdata_locator::get()->symbol_from_string(symbol);

        //string symb = inputdata_locator::get()->string_from_symbol(symbol_nr);
        //cout << symbol << " == " << symb << endl;

        int source_nr = e["source"];
        int target_nr = e["target"];

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
    if(ADD_TAILS){
        t->next_in_list = tails_head;
        tails_head = t;
    }

    if(access_trace == nullptr){
        if(t->past() != nullptr)
            access_trace = inputdata_locator::get()->access_trace(t->past());
        else
            access_trace = mem_store::create_trace();
    }
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
    // keep the old node number
    // number = -1;
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

void apta_node::set_child(tail* t, apta_node* node){
    int symbol = t->get_symbol();
    auto it = guards.lower_bound(symbol);
    auto it_end = guards.upper_bound(symbol);
    for(;it != it_end; ++it){
        if(it->second->bounds_satisfy(t)){
            break;
        }
    }
    if(it != guards.end()){
        if(node != 0)
            it->second->target = node;
        else
            guards.erase(it);
    } else {
        apta_guard* g = new apta_guard();
        guards.insert(std::pair<int,apta_guard*>(t->get_symbol(),g));
        g->target = node;
    }
};

std::set<apta_node*>* apta_node::get_sources(){
    auto* sources = new std::set<apta_node*>();
    sources->insert(find());
    if(source != nullptr){
        sources->insert(source->find());
        for(apta_node* n = representative_of; n != nullptr; n = n->next_merged_node){
            if(n->source != nullptr){
                sources->insert(n->source->find());
            }
        }
    }
    return sources;
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

void apta::set_json_depths(){
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
        apta_node *n = *Ait;
        if (n->source == nullptr){
            n->depth = 0;
        } else {
            std::set<apta_node*>* sources = n->get_sources();
            n->depth = n->source->find()->depth + sources->size();
            delete sources;
        }
    }
    std::set<int> depths;
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