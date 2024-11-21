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

#include "oracle_base.h"
#include "misc/sqldb.h"
#include "source/input/inputdata.h"
#include "sul_base.h"

#include <memory>

class active_learning_main_func {
  private:
    // TODO: remove all those here if possible
    //inputdata* get_inputdata() const;
    //std::ifstream get_inputstream() const;
    //std::unique_ptr<parser> get_parser(std::ifstream& input_stream) const;

  public:
    active_learning_main_func() = default;
    void run_active_learning();
};

#endif
