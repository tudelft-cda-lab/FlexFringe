/**
 * @file factories.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The factories for the main objects: The SUL, the oracle, and the algorithm.
 * @version 0.1
 * @date 2024-11-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_FACTORIES_H_
#define _AL_FACTORIES_H_

#include "algorithm_base.h"

#include "ldot.h"
#include "lsharp.h"
#include "lstar_imat.h"
#include "lstar.h"
#include "paul.h"
#include "probabilistic_lsharp.h"
#include "weighted_lsharp.h"

#include <vector>
#include <string_view>
#include <memory>

/**
 * @brief The algorithm factory. This should be the only factory pattern that actually is accessible from outside of this module.
 * 
 */
class algorithm_factory {
  public:
    algorithm_factory() = delete;
    static std::unique_ptr<algorithm_base> create_algorithm_obj();
};




#endif
