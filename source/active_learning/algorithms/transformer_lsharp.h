/**
 * @file transformer_lsharp.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-11-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _TRANSFORMER_L_SHARP_H_
#define _TRANSFORMER_L_SHARP_H_

#include "weighted_lsharp.h"
#include "lsharp.h"

#include "base_teacher.h"
#include "definitions.h"
#include "eq_oracle_base.h"
#include "inputdata.h"
#include "refinement.h"
#include "state_merger.h"
#include "tail.h"
#include "trace.h"
#include "weight_comparing_oracle.h"

#include <list>
#include <memory>
#include <unordered_map>

class transformer_lsharp_algorithm : public weighted_lsharp_algorithm {
  private:
    bool MAX_DEPTH_REACHED = false;

  protected:
    virtual void query_weights(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                              const std::vector<int>& alphabet,
                              std::optional<active_learning_namespace::pref_suf_t> seq_opt) const;

  public:
    transformer_lsharp_algorithm(std::shared_ptr<sul_base>& sul, std::unique_ptr<base_teacher>& teacher,
                              std::unique_ptr<eq_oracle_base>& oracle)
        : weighted_lsharp_algorithm(sul, teacher, oracle) {
        std::cout << "Probabilistic L# only works with probabilistic oracle. Automatically switched to that one.\
If this is undesired behavior check your input and/or source code."
                  << std::endl;
        this->oracle.reset(new weight_comparing_oracle(sul));
    };

    virtual void run(inputdata& id) override;
};

#endif
