/**
 * @file paul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-08-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _PAUL_H_
#define _PAUL_H_

#include "algorithm_base.h"
#include "base_teacher.h"
#include "definitions.h"
#include "eq_oracle_base.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"
#include "overlap_fill.h"

#include <list>
#include <memory>

class paul_algorithm : public algorithm_base {
  protected:
    std::unique_ptr<ii_base> ii_handler;
    refinement* get_best_refinement(unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta, unique_ptr<base_teacher>& teacher);

  public:
    paul_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                     std::unique_ptr<eq_oracle_base>& oracle)
        : algorithm_base(sul, teacher, oracle){
          
          ii_handler = std::make_unique<overlap_fill>();
        };

    void run(inputdata& id) override;
};

#endif
