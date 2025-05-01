/**
 * @file linear_conflict_search_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_LINEAR_CONFLICT_SEARCH_H_
#define _AL_LINEAR_CONFLICT_SEARCH_H_

#include "conflict_search_base.h"

class linear_conflict_search final : public conflict_search_base {
  protected:
    std::vector<int> current_substring;
    std::vector< std::vector<int> > current_substring_batch_format = std::vector< std::vector<int> >(1);

    inline void update_current_substring(const std::vector<int>& cex) noexcept;
    inline void update_current_substring(const std::vector< std::vector<int> >& cex) noexcept;

    template<typename T> requires (std::is_same_v<T, std::vector<int> > || std::is_same_v<T, std::vector< std::vector<int> > >)
    bool get_creates_conflict(const T& cex, apta& hypothesis, inputdata& id);

    template<typename T> requires (std::is_same_v<T, std::vector<int> > || std::is_same_v<T, std::vector< std::vector<int> > >)
    std::pair< std::vector<int>, sul_response> get_conflict_string_common(const T& cex, apta& hypothesis, inputdata& id);
    
  public:
    linear_conflict_search(const std::shared_ptr<conflict_detector_base>& cd) : conflict_search_base(cd) {};
    std::pair< std::vector<int>, sul_response> get_conflict_string(const std::vector<int>& cex, apta& hypothesis, inputdata& id) override;
    std::pair< std::vector<int>, sul_response> get_conflict_string(const std::vector< std::vector<int> >& cex, apta& hypothesis, inputdata& id) override;
};

/**
 * @brief Simple template to make get_conflict_string more readable via avoiding too many nested if-statements.
 */
template<typename T> requires (std::is_same_v<T, std::vector<int> > || std::is_same_v<T, std::vector< std::vector<int> > >)
bool linear_conflict_search::get_creates_conflict(const T& cex, apta& hypothesis, inputdata& id){
  if constexpr(std::is_same_v<T, std::vector<int> >){
    return conflict_detector->creates_conflict(current_substring, hypothesis, id).first;
  }
  else if constexpr(std::is_same_v<T, std::vector< std::vector<int> > >){
    return conflict_detector->creates_conflict(current_substring_batch_format, hypothesis, id).first;
  }
}

/**
 * @brief Searches for a conflict within a DFA via linear search from beginning to end.
 * 
 * We search for the shortest string that actually causes the error to happen.
 * 
 * @param cex The counterexample.
 * @param hypothesis The merged hypothesis.
 * @param id The inputdata wrapper.
 * @return std::vector<int>, sul_response A vector leading to the conflict including the corresponding SUL response.
 */
template<typename T> requires (std::is_same_v<T, std::vector<int> > || std::is_same_v<T, std::vector< std::vector<int> > >)
std::pair< std::vector<int>, sul_response> linear_conflict_search::get_conflict_string_common(const T& cex, apta& hypothesis, inputdata& id){
  if constexpr(std::is_same_v<T, std::vector<int> >){
    current_substring.clear();
  }
  else if constexpr(std::is_same_v<T, std::vector< std::vector<int> > >){
    current_substring_batch_format.at(0).clear();
  }

  bool resp = false;
  if(AL_TEST_EMTPY_STRING) // IMPORTANT: The underlying oracle also needs to check this
    resp = get_creates_conflict(cex, hypothesis, id);
  else if(!AL_TEST_EMTPY_STRING && cex.empty())
    throw std::invalid_argument("WARNING: al_test_empty_string set to false, but oracle tests empty string. Check your implementation.");
  
  while(!resp){ // works because cex has been determined to lead to conflict already
    update_current_substring(cex);
    resp = get_creates_conflict(cex, hypothesis, id);
  }

  if constexpr(std::is_same_v<T, std::vector<int> >){
    std::optional<sul_response> sul_resp_opt = conflict_detector->creates_conflict(current_substring, hypothesis, id).second; // we assume that the oracle already has ascertained to have found a conflict, therefore this is guaranteed to be one
    return std::make_pair(current_substring, sul_resp_opt.value());
  }
  else if constexpr(std::is_same_v<T, std::vector< std::vector<int> > >){
    std::optional<sul_response> sul_resp_opt = conflict_detector->creates_conflict(current_substring_batch_format, hypothesis, id).second; // we assume that the oracle already has ascertained to have found a conflict, therefore this is guaranteed to be one
    return std::make_pair(current_substring_batch_format.at(0), sul_resp_opt.value());
  }
}

#endif
