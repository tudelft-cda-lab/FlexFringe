/**
 * @file running_mode_factory.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#ifndef __RUNNING_MODE_FACTORY_H_
#define __RUNNING_MODE_FACTORY_H_

#include "running_mode_base.h"

#include <memory>
#include <stdexcept>

/**
 * @brief (Abstract) base class for all the modes that flexfringe can run.
 * 
 */
class running_mode_factory {
  public:
    running_mode_factory() = delete;
    static std::unique_ptr<running_mode_base> get_mode();
};

#endif