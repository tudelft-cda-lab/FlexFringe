/**
 * @file search_strategy_factory.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _CEX_SEARCH_STRATEGY_FACTORY_H_
#define _CEX_SEARCH_STRATEGY_FACTORY_H_

#include "search_base.h"

#include <memory>

/**
 * @brief Only used by oracle. Create a counterexample search strategy object.
 * 
 */
class cex_search_strategy_factory {
  public:
    cex_search_strategy_factory() = delete;
    static std::unique_ptr<search_base> create_search_strategy();
};

#endif // _CEX_SEARCH_STRATEGY_FACTORY_H_