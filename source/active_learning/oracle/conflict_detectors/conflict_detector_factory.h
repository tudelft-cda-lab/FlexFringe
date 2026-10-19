/**
 * @file conflict_detector_factory.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-04-18
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef _CEX_CONFLICT_DETECTOR_FACTORY_H_
#define _CEX_CONFLICT_DETECTOR_FACTORY_H_

#include "conflict_detector_base.h"

#include "distinguishing_sequences_handler_base.h"
#include "sul_base.h"

#include <memory>

/**
 * @brief Only used by oracle. Create a counterexample search strategy object.
 * 
 */
class conflict_detector_factory {
  public:
    conflict_detector_factory() = delete;
    static std::shared_ptr<conflict_detector_base> create_detector(const std::shared_ptr<sul_base>& sul);
    static std::shared_ptr<conflict_detector_base> create_detector(const std::shared_ptr<sul_base>& sul, const std::shared_ptr<distinguishing_sequences_handler_base>& ii_handler);
};

#endif // _CEX_SEARCH_STRATEGY_FACTORY_H_