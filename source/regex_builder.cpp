/**
 * @file regex_builder.cpp
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief For building regex with a DFA.
 * @version 0.1
 * @date 2024-1-9
 *
 * @copyright Copyright (c) 2024
 *
 */


#include "regex_builder.h"
#include "apta.h"
#include "predict.h"
#include "state_merger.h"
#include "input/inputdatalocator.h"
#include "misc/utils.h"
#include <unordered_map>
#include <string>
#include <tuple>
#include <ranges>

regex_builder::regex_builder(apta& the_apta, state_merger& merger, std::tuple<bool, bool, bool>& coloring, std::unordered_map<int, char>& mapping){
    const auto mapping_func = [&mapping](int i) {
        return mapping[i];
    };
    initialize(the_apta, merger, coloring, mapping_func);
}

regex_builder::regex_builder(apta& the_apta, state_merger& merger, std::tuple<bool, bool, bool>& coloring, const std::function<char(int)>& mapping_func){
    initialize(the_apta, merger, coloring, mapping_func);
}

void regex_builder::initialize(apta& the_apta, state_merger& merger, std::tuple<bool, bool, bool>& coloring, const std::function<char(int)>& mapping_func){
    bool include_red = std::get<0>(coloring);
    bool include_blue = std::get<1>(coloring);
    bool include_white = std::get<2>(coloring);
    root = the_apta.get_root();

    // This conversion is heavily inspired from the apta::print_json function.

    // Gather nodes.
    for(merged_APTA_iterator Ait = merged_APTA_iterator(root); *Ait != nullptr; ++Ait) {
        apta_node* n = *Ait;
        if(!include_red && n->red) continue;
        if (!include_white && !n->red) {
            if (n->source != nullptr) {
                if (!n->source->find()->red)
                    continue;
                if (!include_blue)
                    continue;
            }
        }
        states.insert(n);
    }

    LOG_S(INFO) << "Gathered states";

    // Gather transitions.
    for(apta_node* n : states) {

        // Only include transitions if there are transitions to nodes that we include.
        bool found = false;
        for(auto& guard : n->guards){
            if(guard.second->get_target() != nullptr){
                apta_node* target = guard.second->get_target()->find();
                if (!include_white && !target->red) {
                    if (target->source != nullptr) {
                        if (!target->source->find()->red)
                            continue;
                        if (!include_blue)
                            continue;
                    }
                }
                found = true;
                break;
            }
        }
        if(!found) continue; // No transitions to nodes included found.

        // Include ALL transitions of this node.
        // It will include also transitions to nodes we do not have.
        // TODO: Is this a bug in the to_json functionality I copied?

        std::unordered_map<apta_node*, std::string> transition_map;
        for(auto& guard : n->guards){
            auto* target = guard.second->get_target();
            if(target == nullptr) continue;

            apta_node* child = target->find(); // Find representative.

            std::string symbol { mapping_func(guard.first) };
            if (transition_map.contains(child)) {
                transition_map[child] = transition_map[child] + "|" + symbol;
            } else {
                transition_map.insert(std::make_pair(child, symbol));
            }

            r_transitions[child].insert(n);
        }
        transitions.insert(std::make_pair(n, transition_map));
    }

    LOG_S(INFO) << "Gathered transitions";

    // Gather type predictions
    for (apta_node* n : states) {
        auto* tr = n->get_access_trace();
        auto data = get_prediction_mapping(&merger, tr);
        std::string type;
        try {
            type = data["predicted trace type"];
        } catch(std::invalid_argument &e) {
            throw runtime_error("Could not get predicted type, did you set --predicttype 1?");
        }
        if (type.size() == 0) {
            throw runtime_error("Could not get predicted type, did you set --predicttype 1?");
        }
        types_map[type].push_back(n);
    }
#ifndef NDEBUG
    stringstream ss;
    ss << "[";
    for (const auto& [type, nn] : types_map) {
        ss << "(" << type << ": ";
        for (const auto* n : nn) {
            ss << n->get_number() << ", ";
        }
        ss << ")";
    }
    ss << "]";
    std::cout << "Regex from types_map: " << ss.str() << std::endl;
    LOG_S(INFO) << "Regex from types_map: " << ss.str() << std::endl;
#endif

    LOG_S(INFO) << "Gathered predictions";
}

std::string regex_builder::to_regex(int output_state) {
    return to_regex(inputdata_locator::get()->string_from_type(output_state));
}

void regex_builder::print_my_transitions(std::unordered_map<apta_node*, std::unordered_map<apta_node*, std::string>> trans) {
    std::cout << "TRANS" << std::endl;
    for (const auto& x : trans) {
        for (const auto& y : x.second) {
            std::cout << x.first->get_number() << "->" << y.first->get_number() << ":" << y.second << std::endl;
        }
    }
}

std::string regex_builder::to_regex(std::string output_state) {
    // Copy (using copy-assignment) datastructures and manipulate for this output_state.
    std::unordered_set<apta_node*> my_states = states;
    std::unordered_map<apta_node*, std::unordered_set<apta_node*>> my_r_transitions = r_transitions;
    std::unordered_map<apta_node*, std::unordered_map<apta_node*, std::string>> my_transitions = transitions;

    // Add epsilon transitions to final and start node not interacting with the whole.
    auto final_node = make_unique<apta_node>();
    for (auto* n : types_map[output_state]) {
        my_transitions[n].insert(std::make_pair(final_node.get(), ""));
    }
    auto start_node = make_unique<apta_node>();
    my_transitions[start_node.get()].insert(std::make_pair(root, ""));
    my_r_transitions[root].insert(start_node.get());

    // Minimally connected nodes
    // Gather connection size. (degree or valency)
    typedef std::pair<apta_node*, int> DEGREE;
    struct degree_cmp {
        bool operator()(const DEGREE& a, const DEGREE&b) {
            return a.second > b.second;
        }
    };
    std::priority_queue<DEGREE, std::deque<DEGREE>, degree_cmp> min_connected;
    for (apta_node* n : states) {
        min_connected.push(std::make_pair(n, r_transitions[n].size() + transitions[n].size()));
    }

    LOG_S(INFO) << "Initialize setup for regex, entering node elemination loop.";

    // Implementation inspired from:
    // https://github.com/caleb531/automata/blob/c39df7d588164e64a0c090ddf89ab5118ee42e47/automata/fa/gnfa.py#L345

    // But maybe a better implemention is this:
    // https://github.com/devongovett/regexgen/blob/master/src/regex.js

    // Iteratively remove all the nodes between the start node and the final node,
    // starting with the least connected node.
    // This is why 'states' does not contain the start and end node.

    // I implement here an order that first removes the least connected nodes.
    // I don't know if the overhead of tracking the degree of the nodes 
    // outweights the advantage of removing these kind of nodes first.
    // Might need to be tested.
    std::vector<std::pair<apta_node*, apta_node*>> loop_pairs;
    while(!my_states.empty()) {
        apta_node* remove = min_connected.top().first;
        min_connected.pop(); // remove

        // Instead of removing values from the queue, we add new ones
        // and skip them if already processed here.
        // https://stackoverflow.com/questions/649640/how-to-do-an-efficient-priority-update-in-stl-priority-queue
        if (!my_states.contains(remove)) continue;

        loop_pairs.clear();
        for (apta_node* source : my_r_transitions[remove]) {
            if (remove == source) continue; // Do not replace self-loops
            for (apta_node* target : std::views::keys(my_transitions[remove])) {
                if (remove == target) continue; // Do not replace self-loops
                loop_pairs.emplace_back(source, target);
            }
        }

        for (const auto [source, target] : loop_pairs) {
            auto r1 = add_maybe_brackets(my_transitions[source][remove]);
            auto r3 = add_maybe_brackets(my_transitions[remove][target]);

            // self-loop
            std::string r2 = "";
            if (my_transitions[remove].contains(remove)) {
                r2 = my_transitions[remove][remove];
                if (r2.size() == 1) {
                    r2 = r2.append("*");
                } else if (r2.size() > 1) {
                    r2 = "(" + r2 + ")*";
                }
            }

            const std::string replacing = r1 + r2 + r3;

            // existing edge
            std::string r4 = "";
            if (my_transitions[source].contains(target)) {
                r4 = my_transitions[source][target];
                if (r4 == "") {
                    r4 = "?";
                } else if (brackets(r4)) {
                    r4 = "|(" + r4 + ")";
                } else {
                    r4 = "|" + r4;
                }
            }        

            // Update the data structures and set the new regex.
            if (r4 == "?" && replacing.size() > 1) {
                my_transitions[source][target] = "(" + replacing + ")" + r4;
            } else {
                my_transitions[source][target] = replacing + r4;
            }
            my_r_transitions[target].insert(source);
        }

        // Remove 'remove' from all datastructures
        // and update amount of degrees for these nodes.
        my_states.erase(remove);
        for (auto source : my_r_transitions[remove]) {
            my_transitions[source].erase(remove);
            min_connected.push(std::make_pair(source, r_transitions[source].size() + transitions[source].size()));
        }
        for (auto target : utils::map_keys(my_transitions[remove])) {
            my_r_transitions[target].erase(remove);
            min_connected.push(std::make_pair(target, r_transitions[target].size() + transitions[target].size()));
        }
        my_transitions.erase(remove);
        my_r_transitions.erase(remove);
    }

    std:string regex = my_transitions[start_node.get()][final_node.get()];
    if (regex.size() == 0) return ".*";
    return "^(" + regex + ")$";
}


bool regex_builder::brackets(const std::string& regex) {
    int bracket_open = 0;
    for (const char& c : regex) {
        if (c == '(') {
            bracket_open++;
            continue;
        }
        if (c == ')') {
            bracket_open--;
            continue;
        }
        if (c == '|' && bracket_open == 0) {
            return true;
            continue;
        }
    }
    return false;
}

std::string regex_builder::add_maybe_brackets(const std::string& regex) {
    return brackets(regex) ? "(" + regex + ")" : regex;
}
