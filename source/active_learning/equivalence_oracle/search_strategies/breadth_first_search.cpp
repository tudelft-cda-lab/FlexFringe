/**
 * @file breadth_first_search.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-04-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "breadth_first_search.h"

#include <cassert>
#include <iostream>
#include <list>
#include <stack>
#include <utility>

using namespace std;

/**
 * @brief Normal BFS through the alphabet, testing for size. I implemented this using two stacks. This requires us
 * lots of copying. An alternative method would be to instead implement recursively, which comes at the cost of the
 * heap, but that does not matter much if we assume the max depth to be not too extreme.
 * 
 * TODO: We can rewrite this function memory efficient by only storing indices of the current strings, rather than 
 * storing the prefixes.
 *
 * @param sul The SUl.
 * @param id The inputdata.
 * @return optional< const vector<int> > A counterexample if found.
 */
std::optional<std::vector<int>> bfs_strategy::next(const inputdata& id) {
    if (depth == MAX_SEARCH_DEPTH)
        return nullopt;

    static const std::vector<int>& alphabet = id.get_alphabet();

    static std::vector<int>::const_iterator alphabet_it = alphabet.begin();

    if (depth == 0) [[unlikely]] {

        if (alphabet_it == alphabet.end()) {
            cout << "Depth: " << depth << endl;

            ++depth;
            alphabet_it = alphabet.begin();
            return old_search.top();
        }

        old_search.push(std::vector<int>{*alphabet_it});
        ++alphabet_it;
        return old_search.top();
    }

    if (alphabet_it == alphabet.end()) {
        alphabet_it = alphabet.begin();
        old_search.pop();

        if (old_search.empty()) {
            cout << "Depth: " << depth << endl;

            ++depth;
            old_search = std::move(curr_search);
            curr_search = std::stack<std::vector<int>>();
            return old_search.top();
        }

        return curr_search.top();
    }

    const auto& prefix = old_search.top();
    auto new_pref = std::vector<int>(prefix);
    new_pref.push_back(*alphabet_it);
    curr_search.push(std::move(new_pref));

    ++alphabet_it;
    return curr_search.top();
}
