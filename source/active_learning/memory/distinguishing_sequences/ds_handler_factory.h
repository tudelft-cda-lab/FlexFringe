/**
 * @file ds_handler_factory.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-05-10
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "distinguishing_sequences_handler_base.h"

#include <memory>
#include <string_view>


#ifndef _DS_HANDLER_FACTORY_H_
#define _DS_HANDLER_FACTORY_H_

/**
 * @brief Gets the incomplete information handler.
 */
class ds_handler_factory {
  public:
    ds_handler_factory() = delete;
    static std::shared_ptr<distinguishing_sequences_handler_base> create_ds_handler(const std::shared_ptr<sul_base>& sul, std::string_view handler_name);
};

#endif