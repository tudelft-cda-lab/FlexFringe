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

#include <optional>
#include <string>
#include <tuple>
#include <vector>

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
    // id can be used for a unique identifier for the trace.
    const std::optional<int> id_opt = std::nullopt;
    const std::optional<double> double_opt = std::nullopt;

    const std::optional<std::vector<int>> int_vec_opt = std::nullopt;
    const std::optional<std::vector<double>> double_vec_opt = std::nullopt;

  public:
    /** Constructors  */
    explicit sul_response(){};

    explicit sul_response(const bool x) : bool_opt(x){};
    explicit sul_response(const int x) : int_opt(x){};
    explicit sul_response(const int x, const double d) : int_opt(x), double_opt(d){};
    explicit sul_response(const double x) : double_opt(x){};

    // e.g. for collecting a hidden representation of a network
    explicit sul_response(const int x, const std::vector<double>&& v) : int_opt(x), double_vec_opt(std::move(v)){};
    explicit sul_response(const double x, const std::vector<double>&& v)
        : double_opt(x), double_vec_opt(std::move(v)){};

    // setting full information: type, id, trace
    explicit sul_response(const int x, const int y, const std::vector<int>&& v)
        : int_opt(x), id_opt(y), int_vec_opt(std::move(v)){};

    // vector versions. E.g. for batched queries
    explicit sul_response(const std::vector<int>&& v_int) : int_vec_opt(std::move(v_int)){};
    explicit sul_response(const std::vector<double>&& v_float) : double_vec_opt(std::move(v_float)){};
    explicit sul_response(const std::vector<int>&& v_int, const std::vector<double>&& v_float)
        : int_vec_opt(std::move(v_int)), double_vec_opt(std::move(v_float)){};

    /** Getter-functions */
    bool get_bool() const;
    bool GET_BOOL() const noexcept;
    bool has_bool_val() const noexcept;

    int get_int() const;
    int GET_INT() const noexcept;
    bool has_int_val() const noexcept;

    int get_id() const;
    int GET_ID() const noexcept;
    bool has_id_val() const noexcept;

    double get_double() const;
    double GET_DOUBLE() const noexcept;
    bool has_double_val() const noexcept;

    const std::vector<int>& get_int_vec() const;
    const std::vector<int>& GET_INT_VEC() const noexcept;
    bool has_int_vec() const noexcept;

    const std::vector<double>& get_double_vec() const;
    const std::vector<double>& GET_DOUBLE_VEC() const noexcept;
    bool has_double_vec() const noexcept;
};

class sul_base {
  protected:
    std::ifstream get_input_stream() const;

  public:
    sul_base() = default;
    virtual ~sul_base() = default; // making sure that the destructors of derived classes are called

    virtual const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const = 0;
    virtual const sul_response do_query(const std::vector<std::vector<int>>& query_trace, inputdata& id) const {
        throw std::logic_error("batched queries not implemented in this sul-class.");
    }

    /**
     * @brief Initialize the sul class.
     */
    virtual void pre(inputdata& id) = 0;

    virtual void reset() = 0;
};

#endif
