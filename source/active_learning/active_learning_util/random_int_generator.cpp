/**
 * @file random_int_generator.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-08-19
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "random_int_generator.h"

using namespace std;

/**
 * @brief Sets the upper limit 
 * 
 * @param lower 
 * @param upper 
 */
void random_int_generator::set_limits(const int lower, const int upper) noexcept {
  assert( (lower < upper, (void*) "Lower bound is supposed to be smaller than upper bound.", upper ) );
  this->distribution = uniform_int_distribution<>(lower, upper);
}

/**
 * @brief No surprises here.
 * 
 * @return int Random integer.
 */
int random_int_generator::get_random_int() noexcept {
  return distribution(this->generator);
}