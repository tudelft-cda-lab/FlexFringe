/**
 * @file hashfunction.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Implementation of hash function with inner state.
 * @version 0.1
 * @date 2021-03-16
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef _HASH_FUNCTION_
#define _HASH_FUNCTION_

#include <functional>
#include <random>
#include <chrono>

template<typename T>
class HashFunction{
private:
  int seed;
  std::hash<T> objHash;
  std::hash<int> intHash;

public:
  HashFunction() {
    if(RANDOM_INITIALIZATION_SKETCHES != 0){
      std::default_random_engine generator(static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
      std::uniform_int_distribution<int> equal;
      HashFunction(equal(generator));
    }
    else{
      HashFunction(42);
    }
  }
  
  HashFunction(int seed) noexcept { this->seed = seed; }

  inline void setSeed(int seed) noexcept {
    this->seed = seed;
  } 

  /**
   * @brief Hashes an object, returns a size_t object. See https://en.cppreference.com/w/cpp/utility/hash
   * 
   * @param obj The object to hash.
   * @return std::size_t The hashed object.
   */
  inline unsigned int hash(const T& obj) const noexcept { // TODO: look out for int-uint clashes
    std::size_t h1 = objHash(obj);
    std::size_t h2 = intHash(this->seed);

    return static_cast<unsigned int>(h1 ^ (h2 << 1));
  }
};

#endif