/**
 * @file overlap_fill_factory.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-05-10
 * 
 * @copyright Copyright (c) 2025
 * 
 */


#ifndef _OVERLAP_FILL_FACTORY_H_
#define _OVERLAP_FILL_FACTORY_H_

#include "overlap_fill_base.h"

#include <memory>
#include <string_view>

class overlap_fill_factory {
  public:
    overlap_fill_factory() = delete;

    static std::shared_ptr<overlap_fill_base> create_overlap_fill_handler(const std::shared_ptr<sul_base>& sul, std::string_view ii_name);
};

#endif