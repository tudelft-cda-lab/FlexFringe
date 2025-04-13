/**
 * @file distinguishing_sequence_fill_fast.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Keeps track of distinguishing sequences by storing them in raw format. 
 * As the name implies, this version is fast, but consumes more memory at runtime. Only use 
 * when you don't expect the size of the distinguishing sequences to fit into working memory.
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _DISTINGUISHING_SEQUENCES_FILL_FAST_H_
#define _DISTINGUISHING_SEQUENCES_FILL_FAST_H_

#include "distinguishing_sequence_fill.h"
#include "distinguishing_sequences.h"
#include "parameters.h"

#include <memory>
#include <vector>
#include <list>
#include <unordered_set>

/**
 * @brief Works the same as the distinguishing_sequence_fill_fast, but it memoizes the sequences in a list as well. 
 * This way the sequences do not have to be reconstructed in every turn. The trade-off of this is obviously 
 * it requires more working memory.
 */
class distinguishing_sequence_fill_fast : public distinguishing_sequence_fill {
  private:
    struct suffixes_t {
      private:
        std::vector< std::vector<int> > m_suffixes;

      // hash-function taken from https://stackoverflow.com/a/72073933/11956515
      template<typename T> requires (std::is_same_v<T, std::vector<uint32_t>> || std::is_same_v<T, std::list<uint32_t>>)
      std::size_t get_hash(const T& v) const {
        std::size_t seed = v.size();
        for(auto x : v) {
          x = ((x >> 16) ^ x) * 0x45d9f3b;
          x = ((x >> 16) ^ x) * 0x45d9f3b;
          x = (x >> 16) ^ x;
          seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
      }
      
      public:
        const auto& get_suffixes() const { return m_suffixes; }
        int size() const noexcept { return m_suffixes.size(); }

        template<typename T> requires (std::is_same_v<T, std::vector<int>> || std::is_same_v<T, std::list<int>>)
        void add_suffix(const T& seq);

    };

    suffixes_t m_suffixes;

  protected:
    void pre_compute(std::list<int>& suffix, std::unordered_set<apta_node*>& seen_nodes, std::unique_ptr<apta>& aut, apta_node* left, apta_node* right, const int depth) override;

    std::vector<int> predict_node_with_automaton(apta& aut, apta_node* node) override;
    std::vector<int> predict_node_with_sul(apta& aut, apta_node* node) override;

    const int size() const override {return m_suffixes.size();}

  public:
    distinguishing_sequence_fill_fast(const std::shared_ptr<sul_base>& sul) : distinguishing_sequence_fill(sul){};
    void add_suffix(const std::vector<int>& seq) override {m_suffixes.add_suffix(seq);} 
};

/**
 * @brief Adds a suffix to the set of suffixes. Makes sure that duplicates are avoided.
 */
template<typename T> requires (std::is_same_v<T, std::vector<int>> || std::is_same_v<T, std::list<int>>)
void distinguishing_sequence_fill_fast::suffixes_t::add_suffix(const T& seq) {
  static std::unordered_set<size_t> hashed_suffixes;

  std::size_t suf_hash;
  // reinterpret_cast in following dangerous, but we only want to generate hash, therefore ok
  if constexpr(std::is_same_v< T, std::vector<int> >)
    suf_hash = get_hash(reinterpret_cast< const std::vector<uint32_t>& >(seq));
  else if constexpr(std::is_same_v< T, std::list<int> >)
    suf_hash = get_hash(reinterpret_cast< const std::list<uint32_t>& >(seq));

  if(hashed_suffixes.contains(suf_hash))
    return;

  hashed_suffixes.insert(suf_hash);
  if constexpr(std::is_same_v<T, std::vector<int>>){
    m_suffixes.push_back(seq);
  }
  else if constexpr(std::is_same_v<T, std::list<int>>){
    std::vector<int> seq_vector;
    seq_vector.reserve(seq.size());
    seq_vector.insert(seq_vector.end(), seq.begin(), seq.end());
    m_suffixes.push_back(std::move(seq_vector));
  }
}

#endif
