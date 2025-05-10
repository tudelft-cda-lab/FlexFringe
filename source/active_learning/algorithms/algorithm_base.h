/**
 * @file algorithm_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Abstract base class for algorithms. Used for polymorphism reasons.
 * @version 0.1
 * @date 2023-04-07
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _ALGORITHM_BASE_H_
#define _ALGORITHM_BASE_H_

#include "oracle_factory.h"
#include "sul_factory.h"
#include "ds_handler_factory.h"

#include "parameters.h"
#include "inputdata.h"

#include <memory>
#include <utility>
#include <initializer_list>
#include <iostream>
#include <fstream>
#include <type_traits> // used in derived classes

class algorithm_base {
  protected:
    std::unique_ptr<base_oracle> oracle;

    // helper functions to deal with input data
    inputdata* get_inputdata() const;
    std::ifstream get_inputstream() const;
    std::unique_ptr<parser> get_parser(std::ifstream& input_stream) const;

    void init_standard();

  public:
    algorithm_base(){
    };

    virtual void run(inputdata& id) = 0;
};

#endif
