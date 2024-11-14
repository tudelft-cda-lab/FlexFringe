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
#include "oracle_base.h"

#include <memory>

class ii_base {
  protected:
    inline static bool memoized = false; // set to true when memoize

  public:

    /**
     * @brief Pre-computation on a node pair. For example relevant in distinguishing sequence approach, where we 
     * first collect a few distinguishing sequences before starting.
     */
    virtual void pre_compute(std::unique_ptr<apta>& aut, std::unique_ptr<oracle_base>& oracle, apta_node* left, apta_node* right){
      return;
    }

    /**
     * @brief Pre-computation on single node. For example relevant in distinguishing sequence approach, where we 
     * use this to memoize partial results to speed up computation.
     */
    virtual void pre_compute(std::unique_ptr<apta>& aut, std::unique_ptr<oracle_base>& oracle, apta_node* node){
      return;
    }

    /**
     * @brief This function is meant to complement nodes when attempting to merge. Missing information that is needed to do a better merge will be 
     * determined e.g. by the oracle.
     */
    virtual void complement_nodes(std::unique_ptr<apta>& aut, std::unique_ptr<oracle_base>& oracle, apta_node* left, apta_node* right=nullptr) = 0;
    
    /**
     * @brief We use this function similar to complement_nodes. The difference is that 
     */
    virtual bool check_consistency(std::unique_ptr<apta>& aut, std::unique_ptr<oracle_base>& oracle, apta_node* left, apta_node* right);

    /**
     * @brief This function completes a single node with the oracle.
     * 
     * An example use case would be when the train set lacks prefixes to sequences it actually contains. In the APTA those will be unlabelled states 
     * that actually do exist. We want to complete/label those states with the help of an oracle. 
     * 
     * @param node The node. 
     * @param oracle The oracle.
     */
    virtual void complete_node(apta_node* node, std::unique_ptr<apta>& aut, std::unique_ptr<oracle_base>& oracle);

    /**
     * @brief Size relevant for some optimizations.
     */
    virtual const int size() const {return -1;}

    /**
     * @brief Self explainatory.
     */
    virtual bool has_memoized() const noexcept {return memoized;}

    /**
     * @brief Specializations are important.
     * 
     */
    virtual void memoize() noexcept {memoized = true;}
};

#endif