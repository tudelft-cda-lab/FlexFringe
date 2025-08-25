/**
 * @file differencing_mode.h
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _DIFFERENCING_MODE_H_
#define _DIFFERENCING_MODE_H_

#include "running_mode_base.h"

#include "apta.h"
#include "input/inputdata.h"

class differencing_mode : public running_mode_base {
  private:
    apta* the_apta2;
    double difference(apta*, apta*);
    double symmetric_difference(apta*, apta*);

  public:
    differencing_mode(){
      the_apta2 = new apta();
    }

    ~differencing_mode(){
      delete the_apta2;
    }
    
    void initialize() override;
    int run() override;
};

#endif //_DIFFERENCING_MODE_H_
