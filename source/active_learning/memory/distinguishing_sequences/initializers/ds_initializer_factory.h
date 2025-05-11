/**
 * @file ds_initializer_factory.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-03-30
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef _DS_INITIALIZER_FACTORY_H_
#define _DS_INITIALIZER_FACTORY_H_

#include "src/ds_initializer_base.h"

#include <string_view>
#include <memory>

class ds_initializer_factory final {
  public:
    ds_initializer_factory() = delete;
    
    static std::unique_ptr<ds_initializer_base> get_initializer(std::string_view name);
};

#endif