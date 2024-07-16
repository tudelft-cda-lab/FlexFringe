/**
 * @file conflict_search_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_CONFLICT_SEARCH_BASE_H_
#define _AL_CONFLICT_SEARCH_BASE_H_

#include "source/input/inputdata.h"
#include "base_teacher.h"

#include <memory>
#include <vector>
#include <utility>
#include <optional>

/**
 * @brief The type of the response. Helps us to determine the type to return.
 * 
 */
enum class response_type{
  int_response,
  float_response
};

/**
 * @brief Helper struct to make the responses more generic.
 * 
 */
struct response_wrapper {
private:
  const response_type r_type;

  int int_val;
  float float_val;

public:
  response_wrapper(const response_type rt) : r_type(rt){};

  void set_int(const int i) noexcept {
    int_val = i;
  }

  void set_float(const float f) noexcept {
    float_val = f;
  }

  float get_float_response() const {
    if(r_type != response_type::float_response)
      throw std::runtime_error("Typing mismatch in the response wrapper.");
    return float_val;
  }

  int get_int_response() const {
    if(r_type != response_type::int_response)
      throw std::runtime_error("Typing mismatch in the response wrapper.");
    return int_val;
  }
};

class conflict_search_base {
  public:
    conflict_search_base(){};

    virtual std::pair< std::vector<int>, std::optional<response_wrapper> >  get_conflict_string(const std::vector<int>& cex, apta& hypothesis, const std::unique_ptr<base_teacher>& teacher, inputdata& id) = 0;
};

#endif
