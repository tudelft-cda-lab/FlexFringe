/**
 * @file random_string_search.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-04-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _AL_RANDOM_STRING_SEARCH_SEARCH_H_
#define _AL_RANDOM_STRING_SEARCH_SEARCH_H_

#include "random_int_generator.h"
#include "search_base.h"

#include <random>
#include <vector>

class random_string_search : public search_base {
  private:
    int samples_drawn;
    int max_samples;
    int last_lower_bound; // for optimization purposes

    int delay_counter = 0;
    const int delay =
        10000; // if within the last [delay] suggested counterexamples None of them was one, we terminate the algorithm

    random_int_generator length_generator;
    random_int_generator alphabet_sampler;

    std::vector<int> alphabet_vec;

  public:
    random_string_search(const int max_depth) : search_base(max_depth) {
        samples_drawn = 0;
        last_lower_bound = 5;
        length_generator.set_limits(1, max_depth);

        max_samples = 500000;
    };

    virtual std::optional<std::vector<int>> next(const inputdata& id) override;

    __attribute__((always_inline)) inline std::optional<std::vector<int>> next(const inputdata& id,
                                                                               const int lower_bound);

    virtual void reset() noexcept override { delay_counter = 0; }
};

#endif
