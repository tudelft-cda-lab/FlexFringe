/**
 * @file factories.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The factories for the main objects: The SUL, the oracle, and the algorithm.
 * @version 0.1
 * @date 2024-11-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_FACTORIES_H_
#define _AL_FACTORIES_H_

#include "algorithm_base.h"
#include "oracle_base.h"
#include "sul_base.h"

//#include "ldot.h"
#include "lsharp.h"
#include "lstar_imat.h"
#include "lstar.h"
#include "paul.h"
#include "probabilistic_lsharp.h"
#include "weighted_lsharp.h"

#include <vector>
#include <string_view>
#include <memory>

/**
 * @brief The algorithm factory. This should be the only factory pattern that actually is accessible from outside of this module.
 * 
 */
class algorithm_factory {
  private:
    template<typename T>
    static std::unique_ptr<algorithm_base> create_algorithm_obj(T&& oracle);

  public:
    algorithm_factory() = delete;
    static std::unique_ptr<algorithm_base> create_algorithm_obj();
};

/**
 * @brief Gets the system under learning based on the input parameters.
 * 
 */
class sul_factory {
  private:
    static std::shared_ptr<sul_base> select_sul(std::string_view sul_name);
    static std::vector< std::shared_ptr<sul_base> > create_suls();

  public:
    sul_factory() = delete;

    // giving the algorithm factory only access to the create_suls() method
    class sul_key{
      friend class algorithm_factory;
      sul_key() = default;
    };

    /**
     * @brief Using the key-trick to give algorithm factory access to create_suls(), but none of the other
     * private methods. Key-trick taken from https://stackoverflow.com/a/1609505/11956515
     */
    static std::vector< std::shared_ptr<sul_base> > create_suls(const sul_key&& key){ return sul_factory::create_suls(); }
};

/**
 * @brief 
 * 
 */
class oracle_factory {
  private:
    static std::unique_ptr<oracle_base> create_oracle(const std::shared_ptr<sul_base>& sul, std::string_view oracle_name);

  public:
    oracle_factory() = delete;

    // key trick explained in sul factory
    class oracle_key{
      friend class algorithm_factory;
      oracle_key() = default;
    };

    static std::unique_ptr<oracle_base> create_oracle(const std::shared_ptr<sul_base>& sul, std::string_view oracle_name, const oracle_key&& key){ 
      return create_oracle(sul, oracle_name); 
    }
};


/**
 * @brief Templated routine for selecting the appropriate algorithm. Templated so it can take single 
 * arguments as well as the initialization_list path.
 * 
 * If you have trouble understanding the templated T&& structure look up "Perfect forwarding", e.g. 
 * in "Effective Modern C++" by Scott Meyers
 */
template<typename T>
std::unique_ptr<algorithm_base> algorithm_factory::create_algorithm_obj(T&& oracle){
  if(ACTIVE_LEARNING_ALGORITHM=="ldot")
    throw std::invalid_argument("Not implemented yet."); //return make_unique<ldot_algorithm>(std::forward<T>(oracle));
  else if(ACTIVE_LEARNING_ALGORITHM=="lsharp")
    return make_unique<lsharp_algorithm>(std::forward<T>(oracle));
  else if(ACTIVE_LEARNING_ALGORITHM=="lstar_imat")
    return make_unique<lstar_imat_algorithm>(std::forward<T>(oracle));
  else if(ACTIVE_LEARNING_ALGORITHM=="lstar")
    return make_unique<lstar_algorithm>(std::forward<T>(oracle));
  else if(ACTIVE_LEARNING_ALGORITHM=="paul")
    return make_unique<paul_algorithm>(std::forward<T>(oracle));
  else if(ACTIVE_LEARNING_ALGORITHM=="probabilistic_lsharp")
    return make_unique<probabilistic_lsharp_algorithm>(std::forward<T>(oracle));
  else if(ACTIVE_LEARNING_ALGORITHM=="weighted_lsharp")
    return make_unique<weighted_lsharp_algorithm>(std::forward<T>(oracle));
  else
    throw std::invalid_argument("The active learning algorithm name has not been recognized by the algorithm object factory.");
}

#endif