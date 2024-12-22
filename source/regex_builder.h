/**
 * @file regex_builder.h
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief For building regex with a DFA.
 * @version 0.1
 * @date 2024-1-9
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __REGEX_BUILDER_H__
#define __REGEX_BUILDER_H__

#include "apta.h"
#include "state_merger.h"
#include "running_modes/predict_mode.h"

#include <string>
#include <map>
#include <set>
#include <queue>
#include <memory>

/**
 * @brief Builds a class that for every output symbol constructs a regex to mach strings that output that output symbol.
 *
 * Please note that symbols by apta are internally hold as single points.
 * To keep this in the regex, the single points must be converted to single characters.
 * Provide a lambda or a mapping that can convert this. // TODO: override this somehow.
 *
 * This would mean that a regex conversion for how traces are usually encoded will only 
 * work when its alphabet only consists of single letter symbols (ie 0-9).
 * For a larger alphabet, you could for example use the (also limited) ascii conversions
 * from sqldb.cpp. Which increases it to about some more than 50.
 *
 * TODO: You can extend its range with UTF8.
 * See: https://stackoverflow.com/questions/26074090/iterating-through-a-utf-8-string-in-c11
 */
class regex_builder{
public:
    explicit regex_builder(apta& the_apta, state_merger& merger, std::tuple<bool, bool, bool>& coloring, std::map<int, char>& mapping);
    explicit regex_builder(apta& the_apta, state_merger& merger, std::tuple<bool, bool, bool>& coloring, const std::function<char(int)>& mapping_func);

    std::map<std::string, std::vector<apta_node*>> get_types_map() const { return types_map; }

    std::unique_ptr<predict_mode> predict_mode_ptr = std::make_unique<predict_mode>();

    void initialize(apta& the_apta, state_merger& merger, std::tuple<bool, bool, bool>& coloring, const std::function<char(int)>& mapping_func);

    apta_node* root;

    // States of the DFA
    std::set<apta_node*> states;
    // Transitions in the DFA
    std::map<apta_node*, std::map<apta_node*, std::string>> transitions;
    // Reverse transitions in the DFA
    std::map<apta_node*, std::set<apta_node*>> r_transitions;

    // Predicted types of each state
    std::map<std::string, std::vector<apta_node*>> types_map;


    /**
     * @brief Convert the APTA to a regex string.
     *
     * This function converts the structure to of the APTA to a regex.
     * This is inspired by the implementation found here:
     * https://github.com/caleb531/automata/blob/c39df7d588164e64a0c090ddf89ab5118ee42e47/automata/fa/gnfa.py#L345
     *
     * Argument is the external string representation.
     * A second argument parts indicate to split the amount of final states in multiple ones.
     * This results in more but shorter regexes.
     */
    std::string to_regex(const std::string& output_state);
    std::string to_regex(int output_state);
    std::vector<std::string> to_regex(int output_state, size_t parts);
    std::vector<std::string> to_regex(const std::string& output_state, size_t parts);
    std::string to_regex(std::vector<apta_node*> final_nodes);

    /**
     * @brief Check if the regex string needs to bracketed.
     * If a | is not inside existing brackets, it should be brackets.
     */
    bool brackets(const std::string& regex);
    std::string add_maybe_brackets(const std::string& regex);

    void print_my_transitions(std::map<apta_node*, std::map<apta_node*, std::string>> trans);

    regex_builder(){
        predict_mode_ptr->initialize();
        throw std::runtime_error("TODO: make sure in your implementation that predict mode and regex builder point to the same input file here");
    }
};

#endif
