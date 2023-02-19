/**
 * @file space-saving.h
 * @author Robert Baumgartner
 * @brief This file implements Balle et al.'s prefix space saving sketches (Bootstrapping and learning PDFA from data streams, 2012)
 * @version 0.1
 * @date 2020-05-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _SP_SAVE_SKETCH_
#define _SP_SAVE_SKETCH_

#include "parameters.h"
//#include "inputdata.h"
#include "input/tail.h"

#include <vector>
#include <cmath>
#include <numeric>
#include <map>
#include <set>
#include <utility>

#include <iostream>
#include <limits>
#include <cassert>
#include <type_traits>
#include <functional>
#include <algorithm>


class PrefSpSvSketch{
  private:
    // some constants used in the tests
    inline static const double c0 = 384;
    inline static const double c1 = 2./(1. + sqrt(c0));
    inline static const double c2 = pow(1.-c1, 2) / pow(c1+sqrt(c1), 2);

    inline static bool initialized = false;
    inline static double NU;
    inline static int K;

    double size;

    map<int, int> prefixes_to_counts;
    inline static map<int, int> symbol_to_mapping;

    vector<int> prefix_to_ints(tail* prefix) noexcept;
    void store_prefix(const int pref);

    /**
     * @brief Comparator for sorting the list.
     */
    inline static bool compare_frequencies(const pair<int, double>& p1, const pair<int, double>& p2){
      return (p1.second < p2.second);
    }

    
  public:
    PrefSpSvSketch() = delete;
    PrefSpSvSketch(const int K);

    void operator+(PrefSpSvSketch& other);
    void operator-(PrefSpSvSketch& other);
    void operator=(PrefSpSvSketch& other);

    double compute_distance(PrefSpSvSketch& other);
    double compute_prefix_distance(PrefSpSvSketch& other);

    bool test_on_lower_bound(PrefSpSvSketch& other, const double delta);
    double get_upper_bound_estimate(PrefSpSvSketch& other, const double mu_hat, const double delta, const double R, const double k);

    // for experimental purposes, not in original work
    bool hoeffding(const PrefSpSvSketch& other) const;
    double cosine_similarity(const PrefSpSvSketch& other) const;

    void store(tail* prefix);
    map<int, double> retrieve_frequencies() const;

    vector< pair<int, double> > get_most_frequent(const int N) const;

    inline int get_size() const noexcept {
      return static_cast<int>(size);
    }

    inline int get_n_strings() const noexcept {
      return this->prefixes_to_counts.size();
    }
};

/*--------------------------------- Implementations ------------------------------------*/

/**
 * @brief Construct a new Pref Sp Sv Sketch:: Pref Sp Sv Sketch object
 * 
 * @param K Number of objects to store
 * @param L L: expected string length
 */
inline PrefSpSvSketch::PrefSpSvSketch(const int K) {
  this->size = 0.;
  assert(K > 0 && L > 0); // it does not make sense to count less than 1 prefix(es)

  if(!initialized){
    this->K = K;
    cout << "K: " << K << endl;
    NU = 2*EPSILON;
    cout << "Nu: " << NU << endl;
    initialized = true;
  }
}

vector<int> PrefSpSvSketch::prefix_to_ints(tail* prefix) noexcept {
  if(prefix->is_final()){
    return vector<int>{-1};
  }
  
  vector<int> res;
        
  // mapping the sequences to unique values, see e.g. Learning behavioral fingerprints from Netflows using Timed Automata (Pellegrino et al.)
  int code = 0;
  static const double SPACE_SIZE = pow(ALPHABET_SIZE, NSTEPS_SKETCHES);
  double space_size = SPACE_SIZE;

  for(int i = 0; i < NSTEPS_SKETCHES; ++i){
      if(prefix == 0){
          i = NSTEPS_SKETCHES; // kill the inner loop
          break;
      }

      const int symbol = prefix->get_symbol();
      if(symbol == -1){ // termination symbol via flexfringe-convention
          res.push_back(-1); // code cannot become -1
          break;
      }

      if(symbol_to_mapping.count(symbol) == 0) symbol_to_mapping[symbol] = symbol_to_mapping.size() + 1;

      const auto feature_mapping = symbol_to_mapping.at(symbol);
      code = code + static_cast<int>(symbol_to_mapping.at(symbol) * (space_size / ALPHABET_SIZE));
      res.push_back(code);
      space_size = space_size / ALPHABET_SIZE;

      prefix = prefix->future();
  }
  
  return res;
}


/**
 * @brief Stores a single prefix.
 * 
 * @param pref The prefix to store.
 */
void PrefSpSvSketch::store_prefix(const int pref) {
  this->size += 1;

  if(prefixes_to_counts.size() < K){
    auto iter = prefixes_to_counts.find(pref);
    if(iter==prefixes_to_counts.end()){
      prefixes_to_counts.insert({pref, 1});
    }
    else{
      prefixes_to_counts.at(pref) = prefixes_to_counts.at(pref) + 1; 
    }
    return;
  }
  
  auto iter = prefixes_to_counts.find(pref);

  if(iter == prefixes_to_counts.end()){
    // not in map
    int to_erase_pref;
    int min_value = std::numeric_limits<int>::max();
    for (std::map<int,int>::iterator it=prefixes_to_counts.begin(); it!=prefixes_to_counts.end(); ++it){
      if(it->second < min_value){
        to_erase_pref = it->first;
        min_value = it->second;
      }
    }
    
    prefixes_to_counts.erase(to_erase_pref);
    prefixes_to_counts.insert({pref, min_value+1});

  }
  else{
    prefixes_to_counts.at(pref) += 1;
  }
}

/**
 * @brief Store a count of data in data structure.
 * 
 * @param t The prefix (prefix) to store.
 */
void PrefSpSvSketch::store(tail* t) {
  if(t == 0) return;

  const vector<int> prefixes = prefix_to_ints(t);
  for(const auto pref: prefixes){
    store_prefix(pref);
  }
}

/**
 * @brief Get the frequent approximations
 * 
 * @return map<string, double> Pref-frequency pairs
 */
map<int, double> PrefSpSvSketch::retrieve_frequencies() const {
  map<int, double> res;
  for (std::map<int,int>::const_iterator it=prefixes_to_counts.cbegin(); it!=prefixes_to_counts.cend(); ++it){
    const auto& pref = it->first;
    const auto frequency = double(it->second)/this->size;
    res.insert({pref, frequency});
  }

  return res;
}

/**
 * @brief Gets the N most frequent strings. Mostly for visualization, not for the computational part.
 * 
 * @param N integer Number of most frequent n-grams.
 * @return vector< tuple<string, double> > What you think it is.
 */
vector< pair<int, double> > PrefSpSvSketch::get_most_frequent(const int N) const {
  const auto frequencies = this->retrieve_frequencies();
  vector< pair<int, double > > tmp;
  for(const auto& f: frequencies){
    tmp.push_back(f);
  }

  std::sort(tmp.begin(), tmp.end(), compare_frequencies); // sort in ascending order
  std::reverse(tmp.begin(), tmp.end());

  const int UPPER = min(N, int(tmp.size()));
  if(tmp.size()==0) return tmp;

  return vector< pair<int, double> > (tmp.begin(), tmp.begin()+UPPER);
}

/**
 * @brief Performs a Hoeffding-bound test on the two sketches.
 * 
 * @param alpha Parameter for hoeffding-bound.
 */
bool PrefSpSvSketch::hoeffding(const PrefSpSvSketch& other) const {
  double res = 0;
  set<int> processed;

  const double rhs = sqrt(0.5 * log(2 / CHECK_PARAMETER)) * ((1 / sqrt(this->size)) + (1 / sqrt(other.size)));

  const auto& other_sketch = other.prefixes_to_counts;
  const auto other_size = other.size;

  for (std::map<int,int>::const_iterator it=prefixes_to_counts.cbegin(); it!=prefixes_to_counts.cend(); ++it){
    const auto& pref = it->first;

    const double this_f = double(it->second)/this->size;
    double other_f = 0;

    const auto other_entry = other_sketch.find(pref);
    if(other_entry != other_sketch.cend()){
      other_f = static_cast<double>(other_entry->second) / other_size; // TODO: do the counts as double, so you don't have to cast all the time
    }
    const double lhs = abs(this_f - other_f);
    if(lhs >= rhs) return false;

    processed.insert(pref);
  }

  for (std::map<int,int>::const_iterator it=other_sketch.cbegin(); it!=other_sketch.cend(); ++it){
    const auto& other_pref = it->first;
    if(processed.count(other_pref) > 0){
      continue;
    }

    const double other_f = double(it->second)/other.size;
    double this_f = 0;
    
    const auto entry = prefixes_to_counts.find(other_pref);
    if(entry != prefixes_to_counts.cend()){
      this_f = static_cast<double>(entry->second) / this->size; // TODO: do the counts as double, so you don't have to cast all the time
    }

    const double lhs = abs(this_f - other_f);
    if(lhs >= rhs) return false;
  }

  return true;
}

/**
 * @brief For experimental purposes.
 * 
 * @param other Other sketch.
 * @return double What you think it is.
 */
double PrefSpSvSketch::cosine_similarity(const PrefSpSvSketch& other) const {
  double denominator = 0;
  double absolute1 = 0, absolute2 = 0;

  set<int> processed;

  const auto& other_sketch = other.prefixes_to_counts;
  const auto other_size = other.size;

  for (std::map<int,int>::const_iterator it=prefixes_to_counts.cbegin(); it!=prefixes_to_counts.cend(); ++it){
    const auto& pref = it->first;

    const double this_f = double(it->second)/this->size;
    double other_f = 0;

    const auto other_entry = other_sketch.find(pref);
    if(other_entry != other_sketch.cend()){
      other_f = static_cast<double>(other_entry->second) / other_size; // TODO: do the counts as double, so you don't have to cast all the time
    }

    denominator += this_f * other_f;

    absolute1 += pow(this_f, 2);
    absolute2 += pow(other_f, 2);

    processed.insert(pref);
  }

  for (std::map<int,int>::const_iterator it=other_sketch.cbegin(); it!=other_sketch.cend(); ++it){
    const auto& other_pref = it->first;
    if(processed.count(other_pref) > 0){
      continue;
    }

    const double other_f = double(it->second)/other.size;
    double this_f = 0;
    
    const auto entry = prefixes_to_counts.find(other_pref);
    if(entry != prefixes_to_counts.cend()){
      this_f = static_cast<double>(entry->second) / this->size; // TODO: do the counts as double, so you don't have to cast all the time
    }

    denominator += this_f * other_f;

    absolute1 += pow(this_f, 2);
    absolute2 += pow(other_f, 2);
  }

  absolute1 = sqrt(absolute1);
  absolute2 = sqrt(absolute2);

  if(absolute1 == 0 || absolute2 == 0){
    return 0;
  }
    
  return denominator / (absolute1 * absolute2);
}


/**
 * @brief Compute the prefix distance (Lp distance in Balle et al.'s paper) in between two distributions.
 * 
 * @param other The other sketch.
 * @return double The distance. Note: Because frequencies do not necessarily have to overlap, this 
 * quantity is NOT upper-bounded by 1. 
 */
double PrefSpSvSketch::compute_prefix_distance(PrefSpSvSketch& other){
  double res = 0;

  set<int> processed;

  const auto& other_sketch = other.prefixes_to_counts;
  const auto other_size = other.size;

  for (std::map<int,int>::const_iterator it=prefixes_to_counts.cbegin(); it!=prefixes_to_counts.cend(); ++it){
    const auto& pref = it->first;
    const double frequency = double(it->second)/this->size;
    
    const auto other_entry = other_sketch.find(pref);
    if(other_entry == other_sketch.cend()){
      res = max(frequency, res);
    }
    else{
      auto other_f = static_cast<double>(other_entry->second) / other_size; // TODO: do the counts as double, so you don't have to cast all the time
      res = max(abs(frequency - other_f), res);
    }

    processed.insert(pref);
  }

  for (std::map<int,int>::const_iterator it=other_sketch.cbegin(); it!=other_sketch.cend(); ++it){
    const auto& other_pref = it->first;
    if(processed.count(other_pref) > 0){
      continue;
    }

    const double other_frequency = double(it->second)/other.size;
    
    const auto entry = prefixes_to_counts.find(other_pref);
    if(entry == prefixes_to_counts.cend()){
      res = max(other_frequency, res);
    }
    else{
      auto f = static_cast<double>(entry->second) / this->size; // TODO: do the counts as double, so you don't have to cast all the time
      res = max(abs(other_frequency - f), res);
    }
  }

  assert(res < double(2.1)); // 2.1 because of precision errors
  return res;
}


/**
 * @brief Compute the distance in between two distributions.
 * 
 * @param other The other sketch.
 * @return double The distance. Note: Because frequencies do not necessarily have to overlap, this 
 * quantity is NOT upper-bounded by 1. 
 */
double PrefSpSvSketch::compute_distance(PrefSpSvSketch& other){
  double res = 0;

  set<int> processed;

  const auto& other_sketch = other.prefixes_to_counts;
  const auto other_size = other.size;

  for (std::map<int,int>::const_iterator it=prefixes_to_counts.cbegin(); it!=prefixes_to_counts.cend(); ++it){
    const auto& pref = it->first;
    const double frequency = double(it->second)/this->size;
    
    const auto other_entry = other_sketch.find(pref);
    if(other_entry == other_sketch.cend()){
      res += frequency;
    }
    else{
      auto other_f = static_cast<double>(other_entry->second) / other_size; // TODO: do the counts as double, so you don't have to cast all the time
      res += abs(frequency - other_f);
    }

    processed.insert(pref);
  }

  for (std::map<int,int>::const_iterator it=other_sketch.cbegin(); it!=other_sketch.cend(); ++it){
    const auto& other_pref = it->first;
    if(processed.count(other_pref) > 0){
      continue;
    }

    const double other_frequency = double(it->second)/other.size;
    
    const auto entry = prefixes_to_counts.find(other_pref);
    if(entry == prefixes_to_counts.cend()){
      res += other_frequency;
    }
    else{
      auto f = static_cast<double>(entry->second) / this->size; // TODO: do the counts as double, so you don't have to cast all the time
      res += abs(other_frequency - f);
    }
  }

  assert(res < double(2.1)); // 2.1 because of precision errors
  //cout << res << endl;
  return res;
}


/**
 * @brief Perform the VC test according to the paper (Proposition 3). If states are dissimilar return true, unknown == false.
 * 
 * @param other Other sketch.
 * @return bool True if dissimilar, else false for unknown.
 */
bool PrefSpSvSketch::test_on_lower_bound(PrefSpSvSketch& other, const double delta){
  const double mu_hat = compute_prefix_distance(other);
  const double M = (this->size*other.size)/pow(sqrt(this->size) + sqrt(other.size), 2);
  double mu_estimated = mu_hat - NU - sqrt((8/M)*log( 16* ((this->size + other.size)/delta) ));

  //if(mu_estimated > MU){
  if(mu_estimated > 0){ // see paper
    return true; 
  }
  return false;
}

/**
 * @brief This function computes one instance of an upper bound for similarity 
 * according to Corrolary 5 of the paper. 
 * 
 * The minimum has to be computed externally, e.g. in the heuristic itself. 
 * 
 * @param other The other sketch.
 * @param delta Delta.
 * @return float The bound.
 */
double PrefSpSvSketch::get_upper_bound_estimate(PrefSpSvSketch& other, const double mu_hat, const double delta, const double R, const double k){
  const double M = (this->size*other.size)/pow(sqrt(this->size) + sqrt(other.size), 2);
  const double K_k = 4*R*R + 400*pow(this->size + other.size, 2) * pow(R, 5)/(k*k);
  const double confidence_interval = max(sqrt(1/(2*c1*M) * log(K_k/delta)), pow(pow(16*NU, 2)/(2*c2*M) * log(K_k/delta), double(1.)/double(4.)));
  const double estimate = mu_hat + 2*NU + confidence_interval; // TODO: original formula had 8*NU, but the upper bound became really high like this
  //const double estimate = mu_hat + 8*NU + confidence_interval;
  return estimate;
}

/**
 * @brief Add the most frequent contents of the two sketches together.
 * 
 * @param other 
 */
inline void PrefSpSvSketch::operator+(PrefSpSvSketch& other){
  map<int, int> unified_map(this->prefixes_to_counts);

  for(auto& p: other.prefixes_to_counts){
      auto& pref = p.first;
      auto iter = unified_map.find(pref);

      if(iter == unified_map.end()){
        // not in map
        unified_map.insert({pref, p.second});
      }
      else{
        unified_map.at(pref) += p.second;
      }
  }

  this->size += other.size;
  vector< pair<int, double> > all_frequencies;
  
  for(auto& e: unified_map){
    auto frequency = static_cast<double>(e.second) / this->size;
    all_frequencies.push_back({e.first, frequency});
  }

  std::sort(all_frequencies.begin(), all_frequencies.end(), compare_frequencies); // sort in ascending order
  std::reverse(all_frequencies.begin(), all_frequencies.end());

  this->prefixes_to_counts.clear();
  for(int i=0; i<min(K, static_cast<int>(all_frequencies.size())); ++i){
    this->prefixes_to_counts.insert({all_frequencies.at(i).first, unified_map.at(all_frequencies.at(i).first)});
  }
}

/**
 * @brief 
 * 
 * @param other 
 */
inline void PrefSpSvSketch::operator-(PrefSpSvSketch& other){
  throw new runtime_error("Undoing PrefSvSketches not implemented (yet?)."); // not straightforward to implement, we can't just split becuase we discard information when merging
  this->size -= other.size;
}

#endif