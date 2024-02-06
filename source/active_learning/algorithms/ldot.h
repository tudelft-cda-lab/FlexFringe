/**
 * @file ldot.h
 * @author Hielke Walinga (hielkewalinga@gmail.com)
 * @brief The Ldot-algorithm, as part of my thesis
 * @version 0.1
 * @date 2024-1-9
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef _L_DOT_H_
#define _L_DOT_H_

#include "algorithm_base.h"

class ldot_algorithm : public algorithm_base {
  public:
    ldot_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                   std::unique_ptr<eq_oracle_base>& oracle)
        : algorithm_base(sul, teacher, oracle){};

    virtual void run(inputdata& id) override;
};

#endif
