/**
 * @file benchmark_dfaparser.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-04-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "benchmark_dfaparser.h"
#include "common_functions.h"
#include "ctype.h"
#include "mem_store.h"
#include "source/input/inputdatalocator.h"

#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace std;
using namespace graph_information;

/**
 * @brief Does what you think it does.
 *
 * We build the graph successively breadth-first from the root node.
 *
 * @param initial_state Initial state, needed to identify the root node.
 * @param edges state1, list< <state2, label> >.
 * @param nodes state, shape.
 * @return unique_ptr<apta> The sut.
 */
unique_ptr<apta> benchmark_dfaparser::construct_apta(const string_view initial_state,
                                                     const unordered_map<string, list<pair<string, string>>>& edges,
                                                     const unordered_map<string, string>& nodes) const {

    unordered_map<string_view, list<trace*>> node_to_trace_map; // <s2, l> l = list of all traces leading to s2
    unordered_map<string_view, apta_node*> string_to_node_map;

    cout << "Initial state: " << initial_state << "\n\n" << endl;

    cout << "S1: " << endl;
    for (auto& [s1, shape] : nodes) { cout << s1 << endl; }

    cout << "S2: " << endl;

    for (auto& [s2, l] : edges) { cout << s2 << endl; }

    queue<string_view> nodes_to_visit;
    unordered_set<string_view> visited_nodes;
    nodes_to_visit.push(initial_state);

    unique_ptr<apta> sut = unique_ptr<apta>(new apta());
    inputdata* id = inputdata_locator::get();

    apta_node* current_node = mem_store::create_node(nullptr);
    sut->root = current_node;
    int node_number = 0;

    string_to_node_map[initial_state] = sut->root;
    node_to_trace_map[initial_state] = list<trace*>();

    int sequence_nr = 0;
    int depth = 0;
    while (!nodes_to_visit.empty()) {
        string_view s1 = nodes_to_visit.front();
        nodes_to_visit.pop();

        if (visited_nodes.contains(s1))
            continue;

        current_node = string_to_node_map.at(s1);
        current_node->depth = depth;
        current_node->red = true;

        if (!edges.contains(static_cast<string>(s1))) {
            // this is a sink node, it does not have outgoing edges
            continue;
        }

        for (auto& node_label_pair : edges.at(string(s1))) {
            auto& s2 = node_label_pair.first;
            auto& label = node_label_pair.second;

            tail* new_tail = mem_store::create_tail(nullptr);
            const int symbol = id->symbol_from_string(label);
            new_tail->td->symbol = symbol;

            trace* new_trace;
            if (s1 == initial_state) {
                new_trace = mem_store::create_trace(id);
                new_trace->length = 1;

                new_tail->td->index = 0;
                new_trace->head = new_tail;
                new_trace->end_tail = new_tail;
            } else {
                trace* old_trace =
                    node_to_trace_map.at(s1)
                        .front(); // TODO @Sicco: could we safely pick the access trace of the parent node?
                new_trace = mem_store::create_trace(id, old_trace);
                new_trace->length = old_trace->length + 1;
                new_tail->td->index = old_trace->end_tail->td->index + 1;

                tail* old_end_tail = new_trace->end_tail;

                old_end_tail->future_tail = new_tail;
                new_tail->past_tail = old_end_tail;
                new_trace->end_tail = new_tail;
            }

            string_view type_str = nodes.at(static_cast<string>(s2));
            const int type = id->type_from_string(static_cast<string>(type_str));
            new_trace->type = type;
            new_trace->finalize();

            new_trace->sequence = ++sequence_nr;
            new_tail->tr = new_trace;

            apta_node* next_node;
            if (!string_to_node_map.contains(s2)) {
                next_node = mem_store::create_node(nullptr);
                string_to_node_map[s2] = next_node;
                next_node->source = current_node;

                node_to_trace_map[s2] = list<trace*>();
            } else {
                next_node = string_to_node_map[s2];
            }
            current_node->set_child(symbol, next_node);
            current_node->add_tail(new_tail);
            current_node->data->add_tail(new_tail);

            // add the final probs as well
            next_node->add_tail(new_tail->future());
            next_node->data->add_tail(new_tail->future());

            id->add_trace(new_trace);
            node_to_trace_map.at(s2).push_back(new_trace);
            nodes_to_visit.push(s2);
        }
        visited_nodes.insert(s1);
        depth++;
    }

    return std::move(sut);
}