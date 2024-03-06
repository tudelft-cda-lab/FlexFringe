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

template<typename T>
class CountMinSketch{
  private:
    const int width, depth;
    int size;
    std::vector<int> row_sizes;
    std::vector< std::vector<unsigned int> > counts; // counts stored in table

    inline static HashFunction<int> indexHasher;

  public:
    CountMinSketch() = delete;
    CountMinSketch(const int depth, const int width);

    void operator+(const CountMinSketch<T>& other);
    void operator-(const CountMinSketch<T>& other);

    static float sinusSimilarity(const CountMinSketch<T>& c1, const CountMinSketch<T>& c2);
    static float cosineSimilarity(const CountMinSketch<T>& c1, const CountMinSketch<T>& c2);
    static float klDivergence(const CountMinSketch<T>& c1, const CountMinSketch<T>& c2);
    static float chiSquareTest(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2);
    static float zTest(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2);

    static bool hoeffding(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2, const float alpha);
    static bool hoeffdingWithPooling(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2, const float alpha, const int statecount, const int symbolcount, const int correction);
    static float hoeffdingScore(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2, const float alpha);

    void store(const T data, const int row, const int limit) noexcept;
    void storeAt(const int index, const int row) noexcept;
    void printHashes() const noexcept;
    const unsigned int getCount(const T data);

    inline const auto& getSketch() const noexcept {
      return counts;
    }

    inline const auto getWidth() const noexcept {
      return width;
    }

    inline const auto getDepth() const noexcept {
      return depth;
    }

    inline int getSize() const noexcept {
      return size;
    }

    inline int getZeroSize() const noexcept {
      return row_sizes[0];
    }

    /**
     * @brief Reset the counts to zero.
     * 
     */
    inline void reset() noexcept {
      for(auto& row: this->counts){
        for(int i = 0; i < width; ++i){
          row[i] = 0;
        }
      }
    }
};

/*--------------------------------- Implementations ------------------------------------*/

/**
 * @brief Construct a new Count Min Sketch<T>:: Count Min Sketch object
 * 
 * @tparam T The type of the class.
 * @param width The width of the sketch.
 * @param depth The depth of the sketch, which corresponds to the number of hash-functions used.
 */
template<typename T>
inline CountMinSketch<T>::CountMinSketch(const int depth, const int width) : depth(depth), width(width) {
  for(int i = 0; i < depth; ++i){
    counts.push_back(std::vector<unsigned int>(width));
    row_sizes.push_back(0);
  }
  size = 0;
}

/**
 * @brief Store a count of data in data structure.
 * 
 * @tparam T Type of data to be stored.
 * @param data The data to store.
 */
template<typename T>
void CountMinSketch<T>::store(const T data, const int row, const int limit) noexcept {
    const int index = indexHasher.hash(data) % limit;
    counts[row][index] += 1;
    ++size;
    row_sizes[row] += 1;
}

/**
 * @brief Store a count of data in data structure.
 * 
 * @tparam T Type of data to be stored.
 * @param data The data to store.
 */
template<typename T>
void CountMinSketch<T>::storeAt(const int index, const int row) noexcept {
    counts[row][index] += 1;
    ++size;
    row_sizes[row] += 1;
}

/**
 * @brief Retrieve count of data in data structure.
 * 
 * @tparam T Type of data to be stored.
 * @param data The already hashed data.
 */
template<typename T>
inline const unsigned int CountMinSketch<T>::getCount(const T data){
  unsigned int minVal = std::numeric_limits<unsigned int>::max();
  
  for(int i = 0; i < depth; ++i){
    const auto hashedIndex = indexHasher.hash(data) % width; // TODO: not really useful
    minVal = std::min(minVal, counts[i][hashedIndex]);
  }

  return minVal;
}

/**
 * @brief Adds the two count-min-sketches. Can only be done when sizes equal.
 * 
 * @tparam T Type. 
 * @param other The other cms.
 * @return CountMinSketch<T> The result.
 */
template<typename T>
inline void CountMinSketch<T>::operator+(const CountMinSketch<T>& other){
  assert(this->width == other.width && this->depth == other.depth);

	for(int i = 0; i < depth; ++i){
    for(int j = 0; j < width; ++j){
      this->counts[i][j] += other.counts[i][j];
    }
    this->row_sizes[i] += other.row_sizes[i];
  }

  this->size += other.size;
}
 
 /**
 * @brief Subtracts the two count-min-sketches. Can only be done when sizes equal.
 * 
 * @tparam T Type. 
 * @param other The other cms.
 * @return CountMinSketch<T> The result.
 */
template<typename T>
inline void CountMinSketch<T>::operator-(const CountMinSketch<T>& other){
  assert(this->width == other.width && this->depth == other.depth);

	for(int i = 0; i < depth; ++i){
    for(int j = 0; j < width; ++j){
      this->counts[i][j] -= other.counts[i][j];
    }
    this->row_sizes[i] -= other.row_sizes[i];
  }

  this->size -= other.size;
}

/**
 * @brief Computes average cosine similarity among two sketches.
 * 
 * @param c1 Sketch 1.
 * @param c2 Sketch 2.
 * @return float The average similarity.
 */
template <typename T>
inline float CountMinSketch<T>::cosineSimilarity(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2){
  assert(c1.width == c2.width && c1.depth == c2.depth);
  float res = 0;

  for(int i = 0; i < c1.depth; ++i){
    const auto& v1 = c1.counts[i];
    const auto& v2 = c2.counts[i];

    float dot = 0, absolute1 = 0, absolute2 = 0;
    for(int j = 0; j < c2.width; ++j){
      dot += v1[j] * v2[j];
      absolute1 += pow(v1[j], 2);
      absolute2 += pow(v2[j], 2);
    }
    absolute1 = sqrt(absolute1);
    absolute2 = sqrt(absolute2);

    if(absolute1 == 0 && absolute2 == 0){
      //res += 1; // both are zero, but equal
      //continue;
      return 1;
    }
    
    res += dot / (absolute1 * absolute2 + 1e-6);
  }

  return res / c1.depth;
}

/**
 * @brief Computes average sinus similarity among two sketches.
 * 
 * @param c1 Sketch 1.
 * @param c2 Sketch 2.
 * @return float The average similarity.
 */
template <typename T>
inline float CountMinSketch<T>::sinusSimilarity(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2){
  const auto cosine = CountMinSketch<T>::cosineSimilarity(c1, c2);
  return sqrt(1 - cosine * cosine);
}

/**
 * @brief Computes average Kullback-Leiber-Divergence among two sketches.
 * 
 * @param c1 Sketch 1.
 * @param c2 Sketch 2.
 * @return float The average klDivergence.
 */
template <typename T>
inline float CountMinSketch<T>::klDivergence(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2){
  assert(c1.width == c2.width && c1.depth == c2.depth);
  float res = 0.0;

  auto count = 0;
  for(int i = 0; i < c1.depth; ++i){
    const auto& v1 = c1.counts[i];
    const auto& v2 = c2.counts[i];

    const float sum1 = std::accumulate(v1.begin(), v1.end(), 0.0f);
    const float sum2 = std::accumulate(v2.begin(), v2.end(), 0.0f);

    for (int j = 0; j < c1.width; ++j) {
      const float x1 = static_cast<float>(v1[j]) / (sum1 + 1e-3);
      const float x2 = static_cast<float>(v2[j]) / (sum2 + 1e-3);
      if (x1 != 0 && x2 != 0){
        res += x1 * log(x1 / x2);
        ++count;
      }
      else if (x1 != 0 && x2 == 0){
        res += x1 * log(x1 / 1);
        ++count;
      }
    }
  }

  return res / count;
}

/**
 * @brief Performs a chi-square test on the two sketches.
 * 
 * Treat each row of a sketch as a distribution. Then, compute the average of the test on the rows,
 * this is our result. Test for significance needs to be performed externally.
 * 
 * @param c1 Sketch 1.
 * @param c2 Sketch 2.
 * @return float The average similarity.
 */
template <typename T>
inline float CountMinSketch<T>::chiSquareTest(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2){
  assert(c1.width == c2.width && c1.depth == c2.depth);

  double result = 0;

  for(int i = 0; i < c1.depth; ++i){
    // get the sums of the two columns and rows
    std::vector<int> colSum(c1.width);
    double rowSum1 = 0, rowSum2 = 0;
    for(int j = 0; j < c1.width; ++j){
      colSum[j] = c1.counts[i][j] + c2.counts[i][j];
      rowSum1 += c1.counts[i][j];
      rowSum2 += c2.counts[i][j];
    }

    if(rowSum1 == 0 || rowSum2 == 0) return 10e6;

    double chiSquare = 0;
    const double totalSum = rowSum1 + rowSum2 + 10e-3;
    for(int j = 0; j < c1.width; ++j){
      if(colSum[j] == 0) continue;

      colSum[j] = (colSum[j] / totalSum) * 400; // normalize
      double e1 = (colSum[j] * /*rowSum1*/ 200) / 400/*totalSum*/; // normalize
      double e2 = (colSum[j] * /*rowSum2*/ 200) / 400/*totalSum*/; // normalize

      //chiSquare += pow(c1.counts[i][j] - e1, 2) / (e1 + 1e-3);
      //chiSquare += pow(c2.counts[i][j] - e2, 2) / (e2 + 1e-3);
      chiSquare += pow( (c1.counts[i][j] / rowSum1 * 200) - e1, 2) / (e1 + 1e-3); // normalize
      chiSquare += pow( (c2.counts[i][j] / rowSum2 * 200) - e2, 2) / (e2 + 1e-3); // normalize
    }

    result += chiSquare;
  }

  return static_cast<float>(result / c1.depth);
}

/**
 * @brief Performs a z test on the two sketches.
 * 
 * Treat each row of a sketch as a distribution. Then, compute the average of the test on the rows,
 * this is our result. Test for significance needs to be performed externally.
 * 
 * @param c1 Sketch 1.
 * @param c2 Sketch 2.
 * @return float The average similarity.
 */
template <typename T>
inline float CountMinSketch<T>::zTest(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2){
  assert(c1.width == c2.width && c1.depth == c2.depth);

  double z = 0;

  for(int i = 0; i < c1.depth; ++i){
    double m1 = 0, m2 = 0, s1 = 0, s2 = 0; // the means and standard deviations respectively
    int n1 = 0, n2 = 0; // count the samples
    for(int j = 0; j < c1.width; ++j){
      m1 += c1.counts[i][j] * (j + 1);
      m2 += c2.counts[i][j] * (j + 1);

      n1 += c1.counts[i][j];
      n2 += c2.counts[i][j];
    }
    m1 /= n1;
    m2 /= n2;

    // now the deviations
    for(int j = 0; j < c1.width; ++j){
      s1 += pow(c1.counts[i][j] * (j + 1) - m1, 2);
      s2 += pow(c2.counts[i][j] * (j + 1) - m2, 2);
    }
    s1 = sqrt(s1 / (n1 - 1)); // -1 for unbiased estimator
    s2 = sqrt(s2 / (n2 - 1)); // -1 for unbiased estimator

    z += abs(m1 - m2) / sqrt(s1 + s2);
  }

  return static_cast<float>(z / c1.depth);
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
 * @return float The average similarity.
 */
template <typename T>
inline bool CountMinSketch<T>::hoeffding(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2, const float alpha){
  assert(c1.width == c2.width && c1.depth == c2.depth);

  for(int i = 0; i < c1.depth; ++i){

    // count the samples
    int n1 = 0, n2 = 0;
    for(int j = 0; j < c1.width; ++j){
      n1 += c1.counts[i][j];
      n2 += c2.counts[i][j];
    }

    float rhs = sqrt(0.5 * log(2 / alpha)) * ((1 / sqrt(n1)) + (1 / sqrt(n2)));

    // perform the test
    for(int j = 0; j < c1.width; ++j){
      float f1 = c1.counts[i][j];
      float f2 = c2.counts[i][j];

      float lhs = abs((f1 / n1) - (f2 /n2));
      
      if(lhs >= rhs) return false;
    }
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
 * @return float The average similarity.
 */
template <typename T>
inline float CountMinSketch<T>::hoeffdingScore(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2, const float alpha){
  assert(c1.width == c2.width && c1.depth == c2.depth);

  float res = 0;
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
      float f1 = c1.counts[i][j];
      float f2 = c2.counts[i][j];

      float lhs = abs((f1 / n1) - (f2 /n2));
      float rhs = sqrt(0.5 * log(2 / alpha)) * ((1 / sqrt(n1)) + (1 / sqrt(n2)));

      if(lhs >= rhs) return -1;
      res += rhs - lhs;
    }
  }

  return res;
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
 * @param t Upper threshold for pooling. STATE_COUNT in flexfringe
 * @return float The average similarity.
 */
template <typename T>
inline bool CountMinSketch<T>::hoeffdingWithPooling(const CountMinSketch<T>& c1, const CountMinSketch<T>&c2, const float alpha, const int statecount, const int symbolcount, const int correction){
  assert(c1.width == c2.width && c1.depth == c2.depth);
  if(c1.size < statecount || c2.size < statecount) return true;

  // do this for every row of the CMSs separately
  for(int i = 0; i < c1.depth; ++i){

    int divider1 = 0, divider2 = 0;
    int matching2 = 0; // only for c2
    std::vector<float> pool1(2);
    std::vector<float> pool2(2);

    // prepare the pools and dividers
    for(int j = 0; j < c1.width; ++j){
      float f1 = c1.counts[i][j];
      float f2 = c2.counts[i][j];
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
      float f1 = c1.counts[i][j];
      float f2 = c2.counts[i][j];
      if(f1 == 0 || (f1<symbolcount && f2<symbolcount)) continue;

      float lhs = abs(((f1 + correction) / divider1) - ((f2  + correction) /divider2));
      float rhs = sqrt(0.5 * log(2 / alpha)) * ((1 / sqrt(divider1)) + (1 / sqrt(divider2)));

      if(lhs > rhs) return false;
    }

    for(int j = 0; j < 2; ++j){
      float f1 = c1.counts[i][j];
      float f2 = c2.counts[i][j];
      if(f1 == 0 || (f1 < symbolcount && f2 < symbolcount)) continue;

      float lhs = abs(((f1 + correction) / divider1) - ((f2  + correction) /divider2));
      float rhs = sqrt(0.5 * log(2 / alpha)) * ((1 / sqrt(divider1)) + (1 / sqrt(divider2)));

      if(lhs > rhs) return false;
    }
  }

  return true;
}
#endif