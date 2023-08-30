/**
 * @file w_method.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _AL_W_METHOD_SEARCH_H_
#define _AL_W_METHOD_SEARCH_H_

#include "search_base.h"
#include "random_int_generator.h"

#include <random>
#include <vector>

class w_method : public search_base {
  private:
    int samples_drawn;
    int max_samples;
    int last_lower_bound; // for optimization purposes

    int delay_counter = 0;
    const int delay = 1000; // if within the last [delay] suggested counterexamples None of them was one, we terminate the algorithm

    random_int_generator length_generator;
    random_int_generator alphabet_sampler;

    std::vector<int> alphabet_vec;

  public:
    w_method(const int max_depth) : search_base(max_depth) {
      samples_drawn = 0;
      last_lower_bound = 10;

      length_generator.set_limits(last_lower_bound, MAX_SEARCH_DEPTH);

      max_samples = 100000;
    };

    virtual std::optional< std::vector<int> > next(const inputdata& id) override;
    std::optional< std::vector<int> > next(const inputdata& id, const int lower_bound);

    virtual void reset() noexcept override {
      delay_counter = 0;
    }
};

#endif
