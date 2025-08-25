/**
 * @file benchmarkparser_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-04-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "benchmarkparser_base.h"
#include "common_functions.h"
#include "inputdatalocator.h"
#include "parameters.h"

#include <cassert>
#include <fstream>
#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;
using namespace graph_information;

const bool DEBUG = false;

/**
 * @brief Reads a line of the input .dot file.
 *
 * IMPORTANT: Assumptions we make in this function:
 * 1. The last line of the .dot file that contains graph information
 * will only hold the closing braces }
 * 2. States will always be of shape (in order) state_name [shape="shape_type" label="label_name"]
 * 3. Transitions will always be of shape s1 -> s2[label="label_name"]. Note that
 * there is no whitespace in between s2 and the data.
 * 4. Initial transitions will always be prepended by __. That is __start0 [label="" shape="none"]
 * for the data and __start0 -> s1 for the transitions.
 * 5. The first line will just hold the opening of the graph information,
 * but no relevant information about the graph itself.
 *
 * @param input_stream
 * @return unique_ptr<graph_base>
 */
unique_ptr<graph_base> benchmarkparser_base::readline(ifstream& input_stream) const {
    string line, cell;
    std::getline(input_stream, line);

    if (line.empty())
        return unique_ptr<graph_base>(nullptr);

    if (line.rfind('{') != std::string::npos) [[unlikely]] {
        return unique_ptr<graph_base>(new header_line());
    } else if (line.rfind('}') != std::string::npos) [[unlikely]] {
        // test for correct formatting in block
        stringstream final_line(line);
        list<string> final_line_split;
        while (std::getline(final_line, cell, ' ')) final_line_split.push_back(std::move(cell));
        assert(("Last line must be empy but the } character.", final_line_split.size() == 1));

        return unique_ptr<graph_base>(nullptr); // end of file
    }

    stringstream linesplit(line);
    vector<string> cells;
    while (std::getline(linesplit, cell, ' ')) cells.push_back(std::move(cell));

    unique_ptr<graph_base> res;
    if (cells.at(0).at(0) == '_' && cells.at(0).at(1) == '_') {
        // either an initial transition, e.g. " __start0 -> s1 " or a label of it, e.g. " __start0 [label=""
        // shape="none"] "
        if (cells.at(1).compare("->") == 0) {
            res = unique_ptr<initial_transition>(new initial_transition());
            dynamic_cast<initial_transition*>(res.get())->state = std::move(cells.at(2));
            dynamic_cast<initial_transition*>(res.get())->start_id = std::move(cells.at(0));
        } else {
            res = unique_ptr<initial_transition_information>(new initial_transition_information());
            dynamic_cast<initial_transition_information*>(res.get())->start_id = std::move(cells.at(0));

            string& symbol_ref = cells.at(1);
            const auto begin_idx_s = symbol_ref.find_first_of('\"') + 1;
            const auto end_idx_s = symbol_ref.find_last_of('\"');
            if (end_idx_s != begin_idx_s + 1) { // make sure that label is not empty
                const string symbol = symbol_ref.substr(begin_idx_s, end_idx_s - begin_idx_s - 1);
                dynamic_cast<initial_transition_information*>(res.get())->symbol = std::move(symbol);
            }

            // TODO: we'll possibly have to update this one in further cases down the road
            string& data_ref = cells.at(2);
            const auto begin_idx_d = data_ref.find_first_of('\"') + 1;
            const auto end_idx_d = data_ref.find_last_of('\"');
            if (end_idx_d != begin_idx_d + 1) {
                const string data = data_ref.substr(begin_idx_d, end_idx_d - begin_idx_d);
                dynamic_cast<initial_transition_information*>(res.get())->data = std::move(data);
            }
        }
    } else if (cells.at(1).compare("->") == 0) {
        // this is a transition
        res = unique_ptr<transition_element>(new transition_element());
        dynamic_cast<transition_element*>(res.get())->s1 = std::move(cells.at(0));

        string& s2_ref = cells.at(2);
        const auto pos_1 = s2_ref.find_first_of('[');
        const string s2 = s2_ref.substr(0, pos_1);
        dynamic_cast<transition_element*>(res.get())->s2 = std::move(s2);

        // note: no empty labels expected here
        const string label_str = "label=\"";
        const auto label_pos = s2_ref.find(label_str, pos_1 + 1) + label_str.size();
        const auto label_end_pos = s2_ref.find_first_of('\"', label_pos);
        const string label = s2_ref.substr(label_pos, label_end_pos - label_pos);
        dynamic_cast<transition_element*>(res.get())->symbol = std::move(label);

        // TODO: we do not use data at this stage, yet
    } else if (cells.size() == 3) {
        res = unique_ptr<graph_node>(new graph_node());
        dynamic_cast<graph_node*>(res.get())->id = cells.at(0);

        // note: shape is expected to be never empty
        const string shape_str = "shape=\"";
        string& data_ref = cells.at(1);
        const auto shape_pos = data_ref.find(shape_str, 1) + shape_str.size(); // 1 because starts with '['
        const auto shape_end_pos = data_ref.find_first_of('\"', shape_pos);
        const string shape = data_ref.substr(shape_pos, shape_end_pos - shape_pos);
        dynamic_cast<graph_node*>(res.get())->shape = std::move(shape);

        // TODO: we don't use the label of the node at this stage
    } else {
        throw logic_error("Wrong input file? Up until now only DFAs supported.");
    }

    return res;
}

/**
 * @brief Read the input and construct a ready apta from it.
 *
 * TODO: We can possibly split the reading and the construction into two parts, making them more
 * reusable and debugable.
 *
 * @param input_stream
 * @return unique_ptr<apta>
 */
unique_ptr<apta> benchmarkparser_base::read_input(ifstream& input_stream) const {
    string initial_state;                                    // node_id
    unordered_map<string, list<pair<string, string>>> edges; // s1, list< <s2, label> >
    unordered_map<string, string> nodes;                     // s, shape

    unique_ptr<graph_base> line_info = readline(input_stream);
    while (line_info) {
        if (dynamic_cast<transition_element*>(line_info.get()) != nullptr) {
            // inputdata_locator::get()->type_from_string(i);
            // transition_element* li = dynamic_cast<transition_element*>(line_info.get());
            auto li_ptr = dynamic_cast<transition_element*>(line_info.get());
            if (!edges.contains(li_ptr->s1)) {
                edges[li_ptr->s1] = list<pair<string, string>>();
            }
            edges[li_ptr->s1].push_back(make_pair(std::move(li_ptr->s2), std::move(li_ptr->symbol)));
        } else if (dynamic_cast<graph_node*>(line_info.get()) != nullptr) {
            // graph_node* gn = dynamic_cast<graph_node*>(line_info.get());
            auto li_ptr = dynamic_cast<graph_node*>(line_info.get());
            nodes[li_ptr->id] = std::move(li_ptr->shape);
        } else if (dynamic_cast<initial_transition*>(line_info.get()) != nullptr) [[unlikely]] {
            assert(("Only one initial state expected. Wrong input file?", initial_state.size() == 0));
            initial_state = std::move(dynamic_cast<initial_transition*>(line_info.get())->state);
        } else if (dynamic_cast<initial_transition_information*>(line_info.get()) != nullptr) [[unlikely]] {
            // Not needed for DFAs
            // inputdata_locator::get()->symbol_from_string(line_info->symbol);
            // initial_transition_information[line_info->start_id] = make_pair(line_info->symbol, line_info->data);
        } else if (dynamic_cast<header_line*>(line_info.get()) != nullptr) [[unlikely]] {
            // do nothing, but don't continue
        } else {
            throw logic_error("Unexpected object returned. Wrong input file?");
        }
        line_info = readline(input_stream);
    }

    auto sut = construct_apta(initial_state, edges, nodes);

    return std::move(sut);
}