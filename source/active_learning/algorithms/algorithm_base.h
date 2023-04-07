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

#include "inputdata.h"
#include "sul_base.h"

#include <memory>
#include <utility>

class algorithm_base {
  protected:
    std::unique_ptr<sul_base> sul;

  public:
    algorithm_base() = default; // TODO: delete
    algorithm_base(std::unique_ptr<sul_base>& sul) : sul(std::move(sul)){};
    virtual void run(inputdata& id) = 0;
};

#endif