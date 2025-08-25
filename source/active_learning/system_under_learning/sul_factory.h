/**
 * @file sul_factory.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-05-10
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef _SUL_FACTORY_H_
#define _SUL_FACTORY_H_

#include "sul_base.h"

#include <memory>
#include <string_view>

/**
 * @brief Gets the system under learning based on the input parameters.
 * 
 */
class sul_factory {
  public:
    sul_factory() = delete;

    static std::shared_ptr<sul_base> create_sul(std::string_view sul_name);
};

#endif