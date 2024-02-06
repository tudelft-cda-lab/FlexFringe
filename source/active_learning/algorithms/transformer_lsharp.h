/**
 * @file transformer_lsharp.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _TRANSFORMER_L_SHARP_H_
#define _TRANSFORMER_L_SHARP_H_

#include "lsharp.h"
#include "base_teacher.h"
#include "definitions.h"
#include "eq_oracle_base.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"

#include <list>
#include <memory>

class transformer_lsharp_algorithm : public lsharp_algorithm {
  protected:
    //void proc_counterex(const std::unique_ptr<base_teacher>& teacher, inputdata& id, unique_ptr<apta>& hypothesis,
    //                    const std::vector<int>& counterex, std::unique_ptr<state_merger>& merger,
    //                    const refinement_list refs, const vector<int>& alphabet) const;

    virtual void complete_state(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                        const std::vector<int>& alphabet) const override;

    //void update_state(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
    //                  const std::vector<int>& alphabet) const;

  public:
    transformer_lsharp_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                     std::unique_ptr<eq_oracle_base>& oracle)
        : lsharp_algorithm(sul, teacher, oracle){};

    //virtual void run(inputdata& id) override;
};

#endif
