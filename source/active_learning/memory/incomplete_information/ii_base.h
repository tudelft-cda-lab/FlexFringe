/**
 * @file ii_base.h
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

#ifndef __II_BASE_H__
#define __II_BASE_H__

#include "apta.h"
#include "sul_base.h"
#include "common_functions.h" // for derived classes

#include <memory>

class ii_base {
  protected:
    std::shared_ptr<sul_base> sul;

    enum class heuristic_type{
      UNINITIALIZED,
      PAUL_H,
      OTHER
    };

  public:
    ii_base(const std::shared_ptr<sul_base>& sul) : sul(sul){};

    ii_base(){
      throw std::logic_error("Error: ii_base must be equipped with a SUL");
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

    /**
     * @brief Used e.g. by distinguishing sequences. Can be used to set the score of a refinement.
     */
    virtual double get_score(){};

    /**
     * @brief This function completes a single node with the sul.
     * 
     * An example use case would be when the train set lacks prefixes to sequences it actually contains. In the APTA those will be unlabelled states 
     * that actually do exist. We want to complete/label those states with the help of an sul. 
     */
    virtual void complete_node(apta_node* node, std::unique_ptr<apta>& aut);

    /**
     * @brief Size relevant for some optimizations.
     */
    virtual const int size() const {return -1;}

    /**
     * @brief Get a vector of responses starting from the desired node according to the criterion set 
     * by the ii-handler. Predictions are based on the current automaton/hypothesis.
     */
    virtual std::vector<int> predict_node_with_automaton(apta& aut, apta_node* node){
      throw std::invalid_argument("This ii-handler does not support predict_node_with_automaton function");
    }
    
    /**
     * @brief Get a vector of responses starting from the desired node according to the criterion set 
     * by the ii-handler. Predictions are based on sul.
     */
    virtual std::vector<int> predict_node_with_sul(apta& aut, apta_node* node){
      throw std::invalid_argument("This ii-handler does not support predict_node_with_sul function");
    }

    /**
     * @brief A function determining whether the distributions as gained from predict_node_with_automaton
     * and predict_node_with_sul are consistent.
     */
    virtual bool distributions_consistent(const std::vector<int>& v1, const std::vector<int>& v2) {
      throw std::invalid_argument("This ii-handler does not support distributions_consistent function");
    }
};

#endif // __II_BASE_H__