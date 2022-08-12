#include "state_merger.h"
#include "evaluate.h"
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <limits.h>

#include "overlap4logs.h"
#include "parameters.h"

REGISTER_DEF_DATATYPE(overlap4logs_data);
REGISTER_DEF_TYPE(overlap4logs);

// they should be parameters to the program
// (they were in Rick's branch)
long DELAY_COLOR_BOUND1 = 5000;
long DELAY_COLOR_BOUND2 = 10000;

const char* colors[] = { "gray", "green", "orange", "cyan", "red" };
const char* colors_labels[] = { "black", "goldenrod", "crimson" };
const char* colors_nodes[] = { "lightgray", "palegreen", "peachpuff", "lightskyblue1", "lightpink" };
const char* names[] = { "?", "A", "D", "C", "E" };

/*** Data operations ***/
overlap4logs_data::overlap4logs_data() {
    num_type = num_map();
    num_delays = num_long_map();
    trace_ids = set<string>();
};

void overlap4logs_data::store_id(string id) {
   trace_ids.insert(id);
};

void overlap4logs_data::print_state_label(iostream& output){
    if(sink_type()){
        int sum_types = 0;
        for(num_map::iterator it = num_type.begin(); it != num_type.end(); ++it)
            sum_types += it->second;
        output << "[" << sum_types <<"]\n[ ";
        for (int i = 0; i < num_sink_types(); i++)
            output << types(i) << " ";
        output << "]";
    } else {
        int stype = sink_type();
        output << "sink " << names[stype];
    }
};

void overlap4logs_data::print_state_style(iostream& output){
    if(sink_type()){
        int largest_type = 0;
        int count = -1;
        for(num_map::iterator it = num_type.begin(); it != num_type.end(); ++it){
            if(it->second > count){
                largest_type = it->first;
                count = it->second;
            }
        }
        output << " style=filled fillcolor=" << colors_nodes[largest_type] << " ";
    } else {
        int stype = sink_type();
        output << " shape=box style=filled fillcolor=" << colors[stype] << " tooltip=\"";
        for(int i = 0; i < inputdata_locator::get()->get_alphabet_size(); ++i){
            if(node->get_child(i) != 0 && node->get_child(i)->get_data()->sink_type() == stype){
                apta_node* c = node->get_child(i);
                for(set<string>::iterator it3 = reinterpret_cast<overlap4logs_data*>(c->get_data())->trace_ids.begin(); it3 != reinterpret_cast<overlap4logs_data*>(c->get_data())->trace_ids.end(); ++it3){
                    output <<  it3->c_str() << " ";
                }
            }
        }
    }
};

void overlap4logs_data::print_transition_label(iostream& output, int symbol){
    if(node->get_child(symbol) != 0){
        output << " " << inputdata_locator::get()->get_symbol(symbol).c_str() << "\n";

        long minDelay = LONG_MAX;
        long maxDelay = -1;
        double meanDelay = delay_mean(symbol);
        double stdDelay = delay_std(symbol);

        for(long_map::iterator delay_it = num_delays[symbol].begin(); delay_it != num_delays[symbol].end(); delay_it++) {
            if(delay_it->first < minDelay && delay_it->second > 0) {
                minDelay = delay_it->first;
            }
            if(delay_it->first > maxDelay && delay_it->second > 0) {
                maxDelay = delay_it->first;
            }
        }

        output << std::setprecision(0) << "MIN:" << minDelay << " MAX:" << maxDelay;
        output << std::setprecision(1) << " MEAN:" << meanDelay << " STD:" << stdDelay << "\n";
        output << std::setprecision(2);
    }
};

// this should have a pair, set<pair<int, eval_data*>>
void overlap4logs_data::print_transition_style(iostream& output, set<int> symbols){
    apta_node* root = node;
    while(root->get_source() != nullptr) root = root->get_source()->find();
    int root_size = root->get_size();
    int edge_sum = 0;
    for(set<int>::iterator it = symbols.begin(); it != symbols.end(); ++it){
        edge_sum += pos(*it);// old: + neg(*it);
    }
    float penwidth = 0.5 + max(0.1, ((double)edge_sum*10)/(double)root_size);

    int color = 0;

    long minDelay = LONG_MAX;
    long maxDelay = -1;

    for(set<int>::iterator sym_it = symbols.begin(); sym_it != symbols.end(); sym_it++){
        int symbol = *sym_it;
        double meanDelay = delay_mean(symbol);
        for(long_map::iterator delay_it = num_delays[symbol].begin(); delay_it != num_delays[symbol].end(); delay_it++) {
            if(delay_it->first < minDelay && delay_it->second > 0) {
                minDelay = delay_it->first;
            }
            if(delay_it->first > maxDelay && delay_it->second > 0) {
                maxDelay = delay_it->first;
            }
        }
        if (meanDelay >= DELAY_COLOR_BOUND2) {
            color = 2;
            break;
        }
        else if (meanDelay >= DELAY_COLOR_BOUND1) {
            color = 1;
        }
    }

    output << " penwidth=\"" << std::setprecision(1) << penwidth << std::setprecision(2) << "\" color=" << colors_labels[color] << " fontcolor=" << colors_labels[color] << " ";
};

void overlap4logs_data::add_tail(tail *t) {
    overlap_data::add_tail(t);

    // Store the final outcome
    num_type[t->get_type()] = types(t->get_type()) + 1;

    // data is the number of seconds
    if(t->get_data().compare("") == 0) return;

    long delay = std::stol(t->get_data());
    if(num_delays.count(t->get_symbol()) == 0) {
        num_delays[t->get_symbol()] = long_map();
    }
    if(num_delays[t->get_symbol()].count(delay) == 0) {
        num_delays[t->get_symbol()][delay] = 0;
    }
    num_delays[t->get_symbol()][delay] += 1;
};

double overlap4logs_data::delay_mean(int symbol){
    long sum = 0, count = 0;

    for(long_map::iterator delay_it = num_delays[symbol].begin(); delay_it != num_delays[symbol].end(); delay_it++) {
        count += delay_it->second;
        sum += delay_it->first * delay_it->second;
    }

    return (double)sum/(double)count;
}

double overlap4logs_data::delay_std(int symbol){
    long count = 0;
    double mean, standardDeviation = 0.0;

    mean = delay_mean(symbol);

    for(long_map::iterator delay_it = num_delays[symbol].begin(); delay_it != num_delays[symbol].end(); delay_it++) {
        count += delay_it->second;
        for(int i = 0; i < delay_it->second; i++) {
            standardDeviation += pow(delay_it->first - mean, 2);
        }
    }

    return sqrt(standardDeviation / count);
};

void overlap4logs_data::update(evaluation_data* right){
    if(node_type == -1) {
       node_type = right->node_type;
       undo_pointer = right;
       trace_ids.insert(reinterpret_cast<overlap4logs_data*>(right)->trace_ids.begin(), reinterpret_cast<overlap4logs_data*>(right)->trace_ids.end());
    } 
    overlap_data::update(right);
    overlap4logs_data* other = (overlap4logs_data*)right;
    for(num_map::iterator it = other->num_type.begin();it != other->num_type.end(); ++it){
        num_type[it->first] = types(it->first) + it->second;
    }

    for(num_long_map::iterator it2 = other->num_delays.begin(); it2 != other->num_delays.end(); ++it2){
        int symbol = it2->first;
        if(num_delays.count(symbol) == 0) {
            num_delays[symbol] = other->num_delays[symbol];
            continue;
        }

        for(long_map::iterator it3 = other->num_delays[symbol].begin(); it3 != other->num_delays[symbol].end(); ++it3){
            long delay = (*it3).first;

            num_delays[symbol][delay] += (*it3).second;
        }
    }
};

void overlap4logs_data::undo(evaluation_data* right){

    if(right == undo_pointer) {
        for (set<string>::iterator rit = reinterpret_cast<overlap4logs_data*>(right)->trace_ids.begin(); rit != reinterpret_cast<overlap4logs_data*>(right)->trace_ids.end(); ++rit) {
            trace_ids.erase(rit->c_str());
        }
    }
    overlap_data::undo(right);
    overlap4logs_data* other = (overlap4logs_data*)right;
    for(num_map::iterator it = other->num_type.begin();it != other->num_type.end(); ++it){
        num_type[it->first] = types(it->first) - it->second;
    }

    // possible memory leak
    for(num_long_map::iterator it2 = other->num_delays.begin(); it2 != other->num_delays.end(); ++it2){
        int symbol = it2->first;

        for(long_map::iterator it3 = other->num_delays[symbol].begin(); it3 != other->num_delays[symbol].end(); ++it3){
            long delay = (*it3).first;

            num_delays[symbol][delay] -= (*it3).second;
        }
    }
};

/*** Merge consistency ***/
bool overlap4logs::consistent(state_merger *merger, apta_node* left, apta_node* right){
    overlap4logs_data* l = (overlap4logs_data*) left->get_data();
    overlap4logs_data* r = (overlap4logs_data*) right->get_data();

    for(num_map::iterator it = l->num_type.begin();it != l->num_type.end(); ++it){
        int type = it->first;
        int count = it->second;
        if (count > 0 && r->types(type) == 0) {
            inconsistency_found = true;
            return false;
        }
    }

    for(num_map::iterator it = r->num_type.begin();it != r->num_type.end(); ++it){
        int type = it->first;
        int count = it->second;
        if (count > 0 && l->types(type) == 0) {
            inconsistency_found = true;
            return false;
        }
    }
    return true;
};

void overlap4logs::update_score(state_merger *merger, apta_node* left, apta_node* right){
    return count_driven::update_score(merger, left, right);
}

double overlap4logs::compute_score(state_merger* m, apta_node* left, apta_node* right){
    return count_driven::compute_score(m, left, right);
}

/*** Sink logic ***/
int overlap4logs_data::find_end_type(apta_node* node) {
    int endtype = -1;

    // Check for all outgoing transitions if there is _one_ unique final type
    for(guard_map::iterator it = node->guards_start(); it != node->guards_end(); ++it){
        apta_node* n = it->second->get_target();
        if (n == 0) continue;

        for (int i = 0; i < num_sink_types(); i++) {
            if (((overlap4logs_data*) n->get_data())->types(i) > 0) {
                if (endtype == -1) {
                    endtype = i;
                }
                // If the outcome is ambiguous, this should not be a sink node
                else if (endtype != i) {
                    return -1;
                }
            }
        }
    }
    return endtype;
}

int overlap4logs_data::sink_type(){
    if(!USE_SINKS) return -1;

    overlap4logs_data* l = (overlap4logs_data*) node->get_data();

    // For a final node, the type itself is the sink type
    if (l->num_final() > 0 && l->get_final_type() <= num_sink_types()) return l->get_final_type();

    // If we want to consider this as a sink node, make sure it's an unambiguous one
    if (l->num_paths() > STATE_COUNT) return get_path_type();

    return -1;
};

bool overlap4logs_data::sink_consistent(int type){
    if(!USE_SINKS) return false;

    overlap4logs_data* l = (overlap4logs_data*) node->get_data();

    int t = l->get_final_type();
    if(t != -1){
        if(t != type) return false;
        return true;
    }
    if(l->num_paths() > STATE_COUNT){
        t = l->get_path_type();
        if(t != -1){
            if(t != type) return false;
            return true;
        }
    }
    return true;
};

int overlap4logs_data::num_sink_types(){
    if(!USE_SINKS) return 0;
    return 5;
};

/*** Output logic ***/
int overlap4logs::print_labels(iostream& output, apta* aut, overlap4logs_data* data, int symbol) {
    int total = data->pos(symbol); // old: + data->neg(symbol);
    output << " " << inputdata_locator::get()->get_symbol(symbol).c_str() << " (" << total << "=" << (((double)total*100)/(double)aut->get_root()->find()->get_size()) << "%)\n";

    long minDelay = LONG_MAX;
    long maxDelay = -1;
    double meanDelay = data->delay_mean(symbol);
    double stdDelay = data->delay_std(symbol);

    for(long_map::iterator delay_it = data->num_delays[symbol].begin(); delay_it != data->num_delays[symbol].end(); delay_it++) {
        if(delay_it->first < minDelay && delay_it->second > 0) {
            minDelay = delay_it->first;
        }
        if(delay_it->first > maxDelay && delay_it->second > 0) {
            maxDelay = delay_it->first;
        }
    }

    output << std::setprecision(0) << "MIN:" << minDelay << " MAX:" << maxDelay;
    output << std::setprecision(1) << " MEAN:" << meanDelay << " STD:" << stdDelay << "\n";
    output << std::setprecision(2);

    if (meanDelay >= DELAY_COLOR_BOUND2) {
        return 2;
    }
    else if (meanDelay >= DELAY_COLOR_BOUND1) {
        return 1;
    }
    else {
        return 0;
    }
}

