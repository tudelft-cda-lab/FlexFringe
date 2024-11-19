/**
 * @file sul_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base class for the system under learning.
 * @version 0.1
 * @date 2023-02-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _SUL_BASE_H_
#define _SUL_BASE_H_

#include "misc/sqldb.h"
#include "source/input/inputdata.h"

#include <tuple>
#include <string>
#include <vector>
#include <optional>

#include <fstream>
#include <stdexcept>
#include <utility>

/**
 * @brief Wrapper class for the response of a SUL. Enables us to make things more generic.
 * 
 */
struct sul_response {
  private:
    const std::optional<bool> bool_opt = std::nullopt;
    const std::optional<int> int_opt = std::nullopt;
    const std::optional<float> float_opt = std::nullopt;

    const std::optional< std::vector<int> > int_vec_opt = std::nullopt;
    const std::optional< std::vector<float> > float_vec_opt = std::nullopt;

  public:
    /** Constructors  */
    explicit sul_response(const bool x) : bool_opt(x){};
    explicit sul_response(const int x) : int_opt(x){};
    explicit sul_response(const float x) : float_opt(x){};

    // e.g. for batched queries
    explicit sul_response(const vector<int>&& x) : int_vec_opt(std::move(x)){}; 

    // e.g. for collecting a hidden representation of a network
    explicit sul_response(const int x, const std::vector<float>&& v) : int_opt(x), float_vec_opt(std::move(v)){};
    explicit sul_response(const float x, const std::vector<float>&& v) : float_opt(x), float_vec_opt(std::move(v)){};

    // vector versions. E.g. for batched queries
    explicit sul_response(const std::vector<int>&& v_int) : int_vec_opt(std::move(v_int)){};
    explicit sul_response(const std::vector<float>&& v_float) : int_vec_opt(std::move(v_float)){};
    explicit sul_response(const std::vector<int>&& v_int, const std::vector<float>&& v_float) : int_vec_opt(std::move(v_int)), float_vec_opt(std::move(v_float)){};

    /** Getter-functions */
    bool get_bool() const;
    bool GET_BOOL() const noexcept;

    int get_int() const;
    int GET_INT() const noexcept;

    float get_float() const;
    float GET_FLOAT() const noexcept;

    const std::vector<int>& get_int_vec() const;
    const std::vector<int>& GET_INT_VEC() const noexcept;

    const std::vector<float>& get_float_vec() const;
    const std::vector<float>& GET_FLOAT_VEC() const noexcept;
};

class sul_base {
    //friend class teacher_imat;
    //friend class oracle_base;

  protected:
    virtual void reset() = 0;

    std::ifstream get_input_stream() const;

    virtual const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) = 0;
    virtual const sul_response do_query(const std::vector< std::vector<int> >& query_trace, inputdata& id){
      throw std::logic_error("batched queries not implemented in this sul-class.");
    }

  public:
    /**
     * @brief Initialize the sul class.
     */
    virtual void pre(inputdata& id) = 0;
};

#endif
