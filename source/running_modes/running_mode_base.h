/**
 * @file running_mode_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __RUNNING_MODE_BASE_H_
#define __RUNNING_MODE_BASE_H_

#include "state_merger.h"
#include "apta.h"
#include "input/inputdata.h"
#include "evaluate.h"

#include <fstream>
#include <stdexcept>

/**
 * @brief (Abstract) base class for all the modes that flexfringe can run.
 * 
 */
class running_mode_base {
  protected:
    inputdata id;
    apta* the_apta = nullptr;
    evaluation_function* eval= nullptr;
    state_merger* merger = nullptr; // must be set from within constructor

    void read_input_file();

  public:
    ~running_mode_base(){
      if(the_apta != nullptr)
        delete the_apta;
      if(merger != nullptr)
        delete merger;
      //if (input_parser != nullptr)
      //  delete input_parser;
      if (eval != nullptr)
        delete eval;
    }

    
    virtual int run() = 0;
    virtual void initialize();
    virtual void generate_output(){};
};

#endif