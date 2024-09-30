/**
 * @file adaptive_distinguishing_sequences.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The adaptive distinguisinh sequences data structure as e.g. described by Vandraager et al. (2022): "A New
 * Approach for Active Automata Learning Based on Apartness"
 * @version 0.1
 * @date 2023-03-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _ADAPTIVE_DISTINGUISHING_SEQUENCES_H_
#define _ADAPTIVE_DISTINGUISHING_SEQUENCES_H_

#include "distinguishing_sequences.h"

#include <list>
#include <iostream>

class adaptive_distinguishing_sequences : public distinguishing_sequences {
    class ads_node {};

  private:
    ads_node root;

  public:
    adaptive_distinguishing_sequences(){
      std::cerr << "adaptive_distinguishing_sequences not implemented yet. Terminating." << std::endl;
      exit(1); // TODO: Implement
    }
};

#endif