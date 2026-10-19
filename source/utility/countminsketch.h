/**
 * @file countminsketch.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief C++ implementation of Count-Min-Sketch data structure.
 * @version 0.1
 * @date 2021-03-16
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef _COUNT_MIN_SKETCH_
#define _COUNT_MIN_SKETCH_

#include "hashfunction.h"
#include "parameters.h"

#include <vector>
#include <cmath>
#include <numeric>

#include <iostream>
#include <limits>
#include <cassert>
#include <type_traits>
#include <functional>

class CountMinSketch{
  private:
    const int width, depth;
    int size;
    std::vector<int> row_sizes;
    int terminationCounts;

    std::vector< std::vector<unsigned int> > counts; // counts stored in table

    inline static std::vector< HashFunction<int> > indexHasher;
    inline static std::set<int> seenSymbols;

    inline static bool initialized = false;

  public:
    CountMinSketch() = delete;
    CountMinSketch(const int depth, const int width);

    void operator+(const CountMinSketch& other);
    void operator-(const CountMinSketch& other);

    void store(const int symbol) noexcept;
    const unsigned int getCount(const int symbol) const;
    static double cosineSimilarity(const CountMinSketch& c1, const CountMinSketch& c2);
    static double cosineSimilarity(const CountMinSketch& c1, const CountMinSketch& c2, const std::set<int>& symbols);

    const std::vector<double> getDistribution() const;
    const std::vector<double> getDistribution(const std::set<int>& symbols) const;
    const std::pair< std::vector<double>, std::vector<double> > getPooledDistributions(const CountMinSketch& other) const;

    static bool hoeffding(const CountMinSketch& c1, const CountMinSketch&c2);
    static bool hoeffding(const CountMinSketch& c1, const CountMinSketch&c2, const std::set<int>& symbols);
    static bool hoeffdingWithPooling(const CountMinSketch& c1, const CountMinSketch&c2);

    inline const auto& getSketch() const noexcept {
      return counts;
    }

    const auto getWidth() const noexcept {
      return width;
    }

    const auto getDepth() const noexcept {
      return depth;
    }

    int getSize() const noexcept {
      return size;
    }

    const int getFinalCounts() const noexcept {
      return terminationCounts;
    }

    static const int getAlphabetSize() noexcept {
      return seenSymbols.size();
    }
};

/*--------------------------------- Implementations ------------------------------------*/

/**
 * @brief Construct a new Count Min Sketch:: Count Min Sketch object
 * 
 * @param width The width of the sketch.
 * @param depth The depth of the sketch, which corresponds to the number of hash-functions used.
 */
inline CountMinSketch::CountMinSketch(const int depth, const int width) : depth(depth), width(width) {
  for(int i = 0; i < depth; ++i){
    counts.push_back(std::vector<unsigned int>(width)); // INVARIANT: the last row is for the termination symbol
  }
  size = 0;
  terminationCounts = 0;

  if(!initialized){
    int seed = 42;
    for(int i = 0; i < depth; ++i){
      indexHasher.push_back(HashFunction<int>(seed)); // TODO: truly randomize the seed
      seed = seed + 2001;
    }
  }
}

/**
 * @brief Store a count of data in data structure.
 * 
 * @param data The data to store.
 */
void CountMinSketch::store(const int symbol) noexcept {
  ++size;
  if(symbol == -1){
    // we do not want the final symbol to be in seenSymbols, as this would skew the distribution
    this->terminationCounts += 1;
    return;
  }
  for(int row=0; row<depth; ++row){
      const int index = indexHasher[row].hash(symbol) % width;
      counts[row][index] += 1;
  }

  if(this->seenSymbols.count(symbol) == 0){
    this->seenSymbols.insert(symbol);
  }
}

/**
 * @brief Retrieve count of data in data structure.
 * 
 * @param data The already hashed data.
 */
inline const unsigned int CountMinSketch::getCount(const int symbol) const{
  if(symbol==-1){
    return terminationCounts;
  }

  unsigned int minVal = std::numeric_limits<unsigned int>::max();
  
  for(int i = 0; i < depth; ++i){
    const auto hashedIndex = indexHasher[i].hash(symbol) % width; // TODO: not really useful
    minVal = std::min(minVal, counts[i][hashedIndex]);
  }

  return minVal;
}

/**
 * @brief Gets the entire distribution over each ever seen symbol for this sketch.
 * 
 * @return const vector<double> The distribution as a vector.
 */
const std::vector<double> CountMinSketch::getDistribution() const {
  std::vector<double> res;
  for(const auto& symbol: this->seenSymbols){
    const double count = static_cast<double>(getCount(symbol));
    res.push_back(count / this->size);
  }

  res.push_back(static_cast<double>(this->terminationCounts) / this->size);
  return res;
}

const std::vector<double> CountMinSketch::getDistribution(const std::set<int>& symbols) const {
  std::vector<double> res;
  for(const auto& symbol: symbols){
    const double count = static_cast<double>(getCount(symbol));
    res.push_back(count / this->size);
  }

  res.push_back(static_cast<double>(this->terminationCounts) / this->size);
  return res;
}

/**
 * @brief Get a pair of distributions, this time pooled.
 * 
 * @param other The other sketch. We need to know this one to make sure we pool the same counts.
 * @return const pair< vector<double>, vector<double> > Pair of the two distribions, in order <this, other>
 */
const std::pair< std::vector<double>, std::vector<double> > CountMinSketch::getPooledDistributions(const CountMinSketch& other) const {
  std::vector<double> this_d, other_d;
  double this_pool_1 = 0, this_pool_2 = 0;
  double other_pool_1 = 0, other_pool_2 = 0;

  for(const auto& symbol: this->seenSymbols){
    const auto this_count = static_cast<double>(getCount(symbol));
    const auto other_count = static_cast<double>(other.getCount(symbol));

    if(this_count <= SYMBOL_COUNT){
      this_pool_1 += this_count;
      other_pool_1 += other_count;
    }

    if(other_count <= SYMBOL_COUNT){
      this_pool_2 += this_count;
      other_pool_2 += other_count;
    }

    if(this_count > SYMBOL_COUNT && other_count > SYMBOL_COUNT){
      this_d.push_back(this_count / this->size);
      other_d.push_back(other_count / other.size);
    }
  }

  this_d.push_back(this_pool_1 / this->size);
  this_d.push_back(this_pool_2 / this->size);

  other_d.push_back(other_pool_1 / other.size);
  other_d.push_back(other_pool_2 / other.size);

  this_d.push_back(static_cast<double>(this->terminationCounts) / this->size);
  other_d.push_back(static_cast<double>(other.terminationCounts) / other.size);

  return make_pair(this_d, other_d);
}


/**
 * @brief Adds the two count-min-sketches. Can only be done when sizes equal.
 * 
 * @param other The other cms.
 * @return CountMinSketch The result.
 */
inline void CountMinSketch::operator+(const CountMinSketch& other){
  assert(this->width == other.width && this->depth == other.depth);

	for(int i = 0; i < depth; ++i){
    for(int j = 0; j < width; ++j){
      this->counts[i][j] += other.counts[i][j];
    }
  }

  this->terminationCounts += other.terminationCounts;
  this->size += other.size;
}
 
 /**
 * @brief Subtracts the two count-min-sketches. Can only be done when sizes equal.
 * 
 * @param other The other cms.
 * @return CountMinSketch The result.
 */
inline void CountMinSketch::operator-(const CountMinSketch& other){
  assert(this->width == other.width && this->depth == other.depth);

	for(int i = 0; i < depth; ++i){
    for(int j = 0; j < width; ++j){
      this->counts[i][j] -= other.counts[i][j];
    }
  }

  this->terminationCounts -= other.terminationCounts;
  this->size -= other.size;
}

/**
 * @brief Computes average cosine similarity among two sketches.
 * 
 * @param c1 Sketch 1.
 * @param c2 Sketch 2.
 * @return double The average similarity.
 */
inline double CountMinSketch::cosineSimilarity(const CountMinSketch& c1, const CountMinSketch&c2){
  assert(c1.width == c2.width && c1.depth == c2.depth);

  double denominator = 0;
  double absolute1 = 0, absolute2 = 0;

  const auto distribution1 = c1.getDistribution();
  const auto distribution2 = c2.getDistribution();

  for(int i=0; i<distribution1.size(); ++i){
    const auto f1 = distribution1[i];
    const auto f2 = distribution2[i];

    denominator += f1 * f2;

    absolute1 += pow(f1, 2);
    absolute2 += pow(f2, 2);
  }

  absolute1 = sqrt(absolute1);
  absolute2 = sqrt(absolute2);

  return denominator / (absolute1 * absolute2 + 1e-6);
}

inline double CountMinSketch::cosineSimilarity(const CountMinSketch& c1, const CountMinSketch&c2, const std::set<int>& symbols){
  assert(c1.width == c2.width && c1.depth == c2.depth);

  double denominator = 0;
  double absolute1 = 0, absolute2 = 0;

  const auto distribution1 = c1.getDistribution(symbols);
  const auto distribution2 = c2.getDistribution(symbols);

  for(int i=0; i<distribution1.size(); ++i){
    const auto f1 = distribution1[i];
    const auto f2 = distribution2[i];

    denominator += f1 * f2;

    absolute1 += pow(f1, 2);
    absolute2 += pow(f2, 2);
  }

  absolute1 = sqrt(absolute1);
  absolute2 = sqrt(absolute2);

  return denominator / (absolute1 * absolute2 + 1e-6);
}

/**
 * @brief Performs a Hoeffding-bound test on the two sketches.
 * 
 * Treat each row of a sketch as a distribution. Then, compute the average of the test on the rows,
 * this is our result. Test for significance needs to be performed externally.
 * 
 * @param c1 Sketch 1.
 * @param c2 Sketch 2.
 * @param alpha Parameter for hoeffding-bound.
 * @return double The average similarity.
 */
inline bool CountMinSketch::hoeffding(const CountMinSketch& c1, const CountMinSketch&c2){
  assert(c1.width == c2.width && c1.depth == c2.depth);

  const auto distribution1 = c1.getDistribution();
  const auto distribution2 = c2.getDistribution();

  const double n1 = static_cast<double>(c1.size);
  const double n2 = static_cast<double>(c2.size);

  double rhs = sqrt(0.5 * log(2 / CHECK_PARAMETER)) * ((1 / sqrt(n1)) + (1 / sqrt(n2)));

  for(int i=0; i<distribution1.size(); ++i){
    const auto f1 = distribution1[i];
    const auto f2 = distribution2[i];

    double lhs = abs(f1 - f2);
    if(lhs >= rhs) return false;
  }

  return true;
}

inline bool CountMinSketch::hoeffding(const CountMinSketch& c1, const CountMinSketch&c2, const std::set<int>& symbols){
  assert(c1.width == c2.width && c1.depth == c2.depth);

  const auto distribution1 = c1.getDistribution(symbols);
  const auto distribution2 = c2.getDistribution(symbols);

  const double n1 = static_cast<double>(c1.size);
  const double n2 = static_cast<double>(c2.size);

  double rhs = sqrt(0.5 * log(2 / CHECK_PARAMETER)) * ((1 / sqrt(n1)) + (1 / sqrt(n2)));

  for(int i=0; i<distribution1.size(); ++i){
    const auto f1 = distribution1[i];
    const auto f2 = distribution2[i];

    double lhs = abs(f1 - f2);
    if(lhs >= rhs) return false;
  }

  return true;
}

/**
 * @brief Hoeffding bound, but this time with pooling.
 * 
 * @param c1 Sketch 1 
 * @param c2 Sketch 2
 * @return true 
 * @return false 
 */
bool CountMinSketch::hoeffdingWithPooling(const CountMinSketch& c1, const CountMinSketch&c2) {
  assert(c1.width == c2.width && c1.depth == c2.depth);

  const auto dists = c1.getPooledDistributions(c2);

  const auto distribution1 = get<0>(dists);
  const auto distribution2 = get<1>(dists);

  const double n1 = static_cast<double>(c1.size);
  const double n2 = static_cast<double>(c2.size);

  double rhs = sqrt(0.5 * log(2 / CHECK_PARAMETER)) * ((1 / sqrt(n1)) + (1 / sqrt(n2)));

  // TODO: flexfringe has more to offer for pooling than just this
  for(int i=0; i<distribution1.size(); ++i){
    const auto f1 = distribution1[i];
    const auto f2 = distribution2[i];

    double lhs = abs(f1 - f2);
    if(lhs >= rhs) return false;
  }

  return true;
}


/**
 * @brief Performs a Hoeffding-bound test on the two sketches.
 * 
 * Treat each row of a sketch as a distribution. Then, compute the average of the test on the rows,
 * this is our result. Test for significance needs to be performed externally.
 * 
 * @param c1 Sketch 1.
 * @param c2 Sketch 2.
 * @param alpha Parameter for hoeffding-bound.
 * @return double The average similarity.
 */
/* inline double CountMinSketch::hoeffdingScore(const CountMinSketch& c1, const CountMinSketch&c2, const double alpha){
  assert(c1.width == c2.width && c1.depth == c2.depth);

  double res = 0;
  for(int i = 0; i < c1.depth; ++i){

    // count the samples
    int n1 = 0, n2 = 0;
    for(int j = 0; j < c1.width; ++j){
      n1 += c1.counts[i][j];
      n2 += c2.counts[i][j];
    }

    if(n1==0 || n2==0) continue;

    // perform the test
    for(int j = 0; j < c1.width; ++j){
      double f1 = c1.counts[i][j];
      double f2 = c2.counts[i][j];

      double lhs = abs((f1 / n1) - (f2 /n2));
      double rhs = sqrt(0.5 * log(2 / alpha)) * ((1 / sqrt(n1)) + (1 / sqrt(n2)));

      if(lhs >= rhs) return -1;
      res += rhs - lhs;
    }
  }

  return res;
} */

/**
 * @brief Performs a Hoeffding-bound test on the two sketches.
 * 
 * Treat each row of a sketch as a distribution. Then, compute the average of the test on the rows,
 * this is our result. Test for significance needs to be performed externally.
 * 
 * @param c1 Sketch 1.
 * @param c2 Sketch 2.
 * @param alpha Parameter for hoeffding-bound.
 * @param t Upper threshold for pooling. STATE_COUNT in flexfringe
 * @return double The average similarity.
 */
/* inline bool CountMinSketch::hoeffdingWithPooling(const CountMinSketch& c1, const CountMinSketch&c2, const double alpha, const int statecount, const int symbolcount, const int correction){
  assert(c1.width == c2.width && c1.depth == c2.depth);
  if(c1.size < statecount || c2.size < statecount) return true;

  // do this for every row of the CMSs separately
  for(int i = 0; i < c1.depth; ++i){

    int divider1 = 0, divider2 = 0;
    int matching2 = 0; // only for c2
    std::vector<double> pool1(2);
    std::vector<double> pool2(2);

    // prepare the pools and dividers
    for(int j = 0; j < c1.width; ++j){
      double f1 = c1.counts[i][j];
      double f2 = c2.counts[i][j];
      if(f1 == 0) continue;

      matching2 += f2;

      if(f1 >= symbolcount && f2 >= symbolcount){
        divider1 += f1 + correction;
        divider2 += f2 + correction;
      }

      // update pool 1
      if(f2<symbolcount){
        pool1[0] += f1;
        pool2[0] += f2;
      }

      // update pool 2
      if(f1<symbolcount){
        pool1[1] += f1;
        pool2[1] += f2;
      }
    }
    pool2[1] += c2.size - matching2; // TODO: does this work as we want?

    // update the pools
    for(int j = 0; j < 2; ++j){
      if(pool1[j] >= symbolcount || pool2[j] >= symbolcount){
        divider1 += pool1[j] + correction;
        divider2 += pool2[j] + correction;
      }
    }

    if(divider1 < statecount || divider2 < statecount) continue;

    // perform the test
    for(int j = 0; j < c1.width; ++j){
      double f1 = c1.counts[i][j];
      double f2 = c2.counts[i][j];
      if(f1 == 0 || (f1<symbolcount && f2<symbolcount)) continue;

      double lhs = abs(((f1 + correction) / divider1) - ((f2  + correction) /divider2));
      double rhs = sqrt(0.5 * log(2 / alpha)) * ((1 / sqrt(divider1)) + (1 / sqrt(divider2)));

      if(lhs > rhs) return false;
    }

    for(int j = 0; j < 2; ++j){
      double f1 = c1.counts[i][j];
      double f2 = c2.counts[i][j];
      if(f1 == 0 || (f1 < symbolcount && f2 < symbolcount)) continue;

      double lhs = abs(((f1 + correction) / divider1) - ((f2  + correction) /divider2));
      double rhs = sqrt(0.5 * log(2 / alpha)) * ((1 / sqrt(divider1)) + (1 / sqrt(divider2)));

      if(lhs > rhs) return false;
    }
  }

  return true;
}  */
#endif
