/**
 * @file random_int_generators.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-08-19
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _AL_RNG_H_
#define _AL_RNG_H_

#include <random>
#include <cassert>

class random_int_generator {
  private:
    std::random_device rd;
    std::mt19937 generator;
    std::uniform_int_distribution<> distribution;

  public:
    random_int_generator(){
      generator = std::mt19937(rd()); // mersenne_twister_engine seeded with rd()
      distribution = std::uniform_int_distribution<>();
    }

    random_int_generator(const int lower, const int upper){
      assert( (lower < upper, (void*) "Lower bound is supposed to be smaller than upper bound.", upper ) );
      generator = std::mt19937(rd()); // mersenne_twister_engine seeded with rd()
      distribution = std::uniform_int_distribution<>(lower, upper);
    }

    void set_limits(const int lower, const int upper) noexcept;
    int get_random_int() noexcept;
};

#endif