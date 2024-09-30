/**
 * @file ii_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base class for incomplete information module. Name might change in future versions.
 * 
 * The main purpose of this module is to incorporate active learning strategies into passive learning. 
 * For example, when data is missing from passive learning such as missing sequences in the train-set
 * we can ask an oracle for the missing sequence.
 * 
 * @version 0.1
 * @date 2024-08-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __II_BASE_H__
#define __II_BASE_H__

#include "apta.h"
#include "base_teacher.h"

#include <memory>

class ii_base {
  protected:

  public:

    /**
     * @brief Pre-computation before we complement the two nodes. For example relevant in distinguishing sequence approach, where we 
     * first collect a few distinguishing sequences before starting.
     * 
     * @param aut The apta.
     * @param teacher The teacher.
     * @param left The left node.
     * @param right The right node.
     */
    virtual void pre_compute(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right){
      return;
    }

    /**
     * @brief This function is meant to complement nodes when attempting to merge. Missing information that is needed to do a better merge will be 
     * determined e.g. by the teacher.
     */
    virtual void complement_nodes(std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher, apta_node* left, apta_node* right=nullptr) = 0;
    
    /**
     * @brief This function completes a single node with the teacher.
     * 
     * An example use case would be when the train set lacks prefixes to sequences it actually contains. In the APTA those will be unlabelled states 
     * that actually do exist. We want to complete/label those states with the help of an oracle. 
     * 
     * @param node The node. 
     * @param teacher The teacher.
     */
    virtual void complete_node(apta_node* node, std::unique_ptr<apta>& aut, std::unique_ptr<base_teacher>& teacher);


};

#endif