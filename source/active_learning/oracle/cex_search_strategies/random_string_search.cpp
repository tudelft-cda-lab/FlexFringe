/**
 * @file random_string_search.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-08-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "random_string_search.h"

#include <cassert>
#include <iostream>
#include <list>
#include <vector>

using namespace std;

optional<vector<int>> random_string_search::next(const inputdata& id) {
    static int lower_b = 5, upper_b = lower_b + 5;
    static int c = 1;

    if (c % 50000 == 0 && upper_b < MAX_SEARCH_DEPTH) {
        lower_b += 5;
        upper_b = lower_b + 5;
        length_generator.set_limits(last_lower_bound, MAX_SEARCH_DEPTH);

        cout << "Cex search from " << lower_b << " to " << upper_b << endl;
    }
    ++c;

    return next(id, 0);
}

optional<vector<int>> random_string_search::next(const inputdata& id, const int lower_bound) {
    if (samples_drawn % 500 == 0)
        cout << "[" << samples_drawn << "/" << max_samples << "] samples suggested." << endl;
    if (samples_drawn == max_samples) {
        cout << "Exhausted the counterexample search. Wrapping up algorithm." << endl;
        return nullopt;
    } else if (delay_counter == delay) {
        cout << "No counterexample found for " << delay << " rounds. Wrapping up algorithm." << endl;
        return nullopt;
    }

    static bool initialized = false;
    if (!initialized) {
        alphabet_vec = id.get_alphabet();
        alphabet_sampler.set_limits(0, alphabet_vec.size() - 1);

        initialized = true;
    }

    if (lower_bound > last_lower_bound) {
        last_lower_bound = lower_bound;
        length_generator.set_limits(lower_bound, MAX_SEARCH_DEPTH);
    }

    const int output_string_length = length_generator.get_random_int();
    vector<int> res(output_string_length);
    for (int i = 0; i < output_string_length; ++i) { res[i] = alphabet_vec[alphabet_sampler.get_random_int()]; }
    ++samples_drawn;
    ++delay_counter;
    return res;
}