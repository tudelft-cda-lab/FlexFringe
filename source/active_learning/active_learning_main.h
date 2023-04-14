/**
 * @file active_learning_main.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The class maintaining the main subroutine that is starting the active learning.
 * @version 0.1
 * @date 2023-03-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _ACTIVE_LEARNING_MAIN_H_
#define _ACTIVE_LEARNING_MAIN_H_

#include "source/input/inputdata.h"
#include "sul_base.h"
#include "base_teacher.h"
#include "eq_oracle_base.h"

#include <memory>
class active_learning_main_func{
  private:
    inputdata get_inputdata() const;
    
    std::shared_ptr<sul_base> select_sul_class() const;
    std::unique_ptr<base_teacher> select_teacher_class(std::shared_ptr<sul_base>& sul) const;
    std::unique_ptr<eq_oracle_base> select_oracle_class(std::shared_ptr<sul_base>& sul) const;
  
  public:
    active_learning_main_func() = default;
    void run_active_learning();
};

#endif