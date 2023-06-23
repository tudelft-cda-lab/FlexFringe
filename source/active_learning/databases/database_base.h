/**
 * @file database_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-06-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef DATABASE_BASE
#define DATABASE_BASE

#include <fstream>

class database_base {
  protected:
    virtual void initialize(std::ifstream& input) = 0;
  public:
    database_base(std::ifstream& input){
      initialize(input);
    }
};

#endif