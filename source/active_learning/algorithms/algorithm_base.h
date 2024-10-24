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

#include "base_teacher.h"
#include "eq_oracle_base.h"
#include "inputdata.h"
#include "sul_base.h"

#include <memory>
#include <utility>

class algorithm_base {
  protected:
    std::shared_ptr<sul_base> sul;
    std::unique_ptr<base_teacher> teacher;
    std::unique_ptr<eq_oracle_base> oracle;

  public:
    algorithm_base() = default; // TODO: delete
    algorithm_base(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                   std::unique_ptr<eq_oracle_base>& oracle)
        : sul(sul), teacher(std::move(teacher)), oracle(std::move(oracle)){};

    virtual void run(inputdata& id) = 0;
};

#endif