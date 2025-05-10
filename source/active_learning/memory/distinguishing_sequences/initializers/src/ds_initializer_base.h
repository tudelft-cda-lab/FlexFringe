/**
 * @file ds_initializer_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-03-30
 * 
 * @copyright Copyright (c) 2025
 * 
 */
 
#ifndef _DS_INITIALIZER_BASE_H_
#define _DS_INITIALIZER_BASE_H_

#include "apta.h"
#include "../../distinguishing_sequences_handler.h"

#include <memory>

/**
 * @brief Initializer base class only for distinguishing sequences.
 * 
 */
class ds_initializer_base {
  public:
    virtual void init(std::shared_ptr<distinguishing_sequences_handler> ii_handler, std::unique_ptr<apta>& aut){};
};

#endif