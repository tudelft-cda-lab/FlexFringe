/**
 * @file factories.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2024-11-13
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "algorithm_factory.h"

// DFAs
#include "lsharp.h"
#include "lstar.h"
#include "lstar_imat.h"

// weighted machines
#include "probabilistic_lsharp.h"
#include "weighted_lsharp.h"

// databases etc.
#include "ldot.h"
#include "paul.h"

// the oracles
#include "base_oracle.h"
#include "input_file_oracle.h"
#include "paul_oracle.h"

// parsers and input-data representations
#include "abbadingoparser.h"
#include "common.h"
#include "csvparser.h"
#include "input/abbadingoreader.h"
#include "inputdata.h"
#include "inputdatalocator.h"
#include "parameters.h"

#include <initializer_list>

using namespace std;

/**
 * @brief Templated routine for selecting the appropriate algorithm. Templated so it can take single
 * arguments as well as the initialization_list path.
 *
 * If you have trouble understanding the templated T&& structure look up "Perfect forwarding", e.g.
 * in "Effective Modern C++" by Scott Meyers
 */
std::unique_ptr<algorithm_base> algorithm_factory::create_algorithm_obj() {
    if (ACTIVE_LEARNING_ALGORITHM == "ldot")
        return make_unique<ldot_algorithm>();
    else if (ACTIVE_LEARNING_ALGORITHM == "lsharp")
        return make_unique<lsharp_algorithm>();
    else if (ACTIVE_LEARNING_ALGORITHM == "lstar_imat")
        return make_unique<lstar_imat_algorithm>();
    else if (ACTIVE_LEARNING_ALGORITHM == "lstar")
        return make_unique<lstar_algorithm>();
    else if (ACTIVE_LEARNING_ALGORITHM == "paul")
        return make_unique<paul_algorithm>();
    else if (ACTIVE_LEARNING_ALGORITHM == "probabilistic_lsharp")
        return make_unique<probabilistic_lsharp_algorithm>();
    else if (ACTIVE_LEARNING_ALGORITHM == "weighted_lsharp")
        return make_unique<weighted_lsharp_algorithm>();
    else
        throw std::invalid_argument(
            "The active learning algorithm name has not been recognized by the algorithm object factory.");
}
