/**
 * @file overlap_fill_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base class for incomplete information module. Name might change in future versions.
 * 
 * The main purpose of this module is to incorporate active learning strategies into passive learning. 
 * For example, when data is missing from passive learning such as missing sequences in the train-set
 * we can ask an sul for the missing sequence.
 * 
 * @version 0.1
 * @date 2024-08-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __OVERLAP_FILL_BASE_H__
#define __OVERLAP_FILL_BASE_H__

#include "apta.h"
#include "sul_base.h"
#include "common_functions.h" // for derived classes

#include <memory>

class overlap_fill_base {
  protected:
    std::shared_ptr<sul_base> sul;

    enum class heuristic_type{
      UNINITIALIZED,
      PAUL_H,
      OTHER
    };

  public:
    overlap_fill_base(const std::shared_ptr<sul_base>& sul) : sul(sul){};

    overlap_fill_base(){
      throw std::logic_error("Error: overlap_fill_base must be equipped with a SUL");
    }

    virtual void initialize(std::unique_ptr<apta>& aut){
      std::cout << "This ii-handler does not need initialization, or it is not implemented yet." << std::endl;
    }

    /**
     * @brief Pre-computation on a node pair. For example relevant in distinguishing sequence approach, where we 
     * first collect a few distinguishing sequences before starting.
     */
    virtual void pre_compute(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) = 0;

    /**
     * @brief Pre-computation on single node. For example relevant in distinguishing sequence approach, where we 
     * use this to memoize partial results to speed up computation.
     */
    virtual void pre_compute(std::unique_ptr<apta>& aut, apta_node* node) = 0;
    
    /**
     * @brief We use this function similar to complement_nodes. The difference is that it does not add data to the tree.
     */
    virtual bool check_consistency(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) = 0;
};

#endif // __II_BASE_H__