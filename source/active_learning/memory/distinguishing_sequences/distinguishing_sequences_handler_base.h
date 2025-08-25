/**
 * @file distinguishing_sequences_handler_base.h
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

#ifndef __DISTINGUISHING_SEQUENCES_HANDLER_BASE_H__
#define __DISTINGUISHING_SEQUENCES_HANDLER_BASE_H__

#include "apta.h"
#include "sul_base.h"
#include "common_functions.h" // for derived classes

#include <memory>
#include <map>
#include <unordered_map>
#include <ranges>
#include <functional>

class distinguishing_sequences_handler_base {
  protected:
    std::shared_ptr<sul_base> sul;

    enum class heuristic_type{
      UNINITIALIZED,
      PAUL_H,
      OTHER
    };

    struct layerwise_suffixes_t {
      using length_suffixes_map = std::unordered_map<int, std::vector< std::vector<int> >>;

      private:
        length_suffixes_map m_suffixes;

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
        const auto& get_suffixes(const int length) { return m_suffixes[length]; }
        
        int size() const noexcept { 
          int res = 0;
          for(const auto& s : m_suffixes | std::views::values) 
            res += s.size();
          return res; 
        }

        template<typename T> requires (std::is_same_v<T, std::vector<int>> || std::is_same_v<T, std::list<int>>)
        void add_suffix(const T& seq);
    };

  public:
    using layer_predictions_map = std::unordered_map< int, std::vector<int> >;

    #ifdef __FLEXFRINGE_CUDA
      // an aggregate to structure values
      struct device_vector {
          std::unordered_map<int, int*> len_pred_map_d; // maps to device pointers
          std::unordered_map<int, size_t > len_size_map; // maps to size
      };
    #endif

    distinguishing_sequences_handler_base(const std::shared_ptr<sul_base>& sul) : sul(sul){};

    distinguishing_sequences_handler_base(){
      throw std::logic_error("Error: distinguishing_sequences_handler_base must be equipped with a SUL");
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
     * @brief Pre-computation on single node. We use this to memoize partial results to speed up computation.
     */
    virtual void pre_compute(std::unique_ptr<apta>& aut, apta_node* node) = 0;
    
    /**
     * @brief We use this function similar to complement_nodes. The difference is that it does not add data to the tree.
     */
    virtual bool check_consistency(std::unique_ptr<apta>& aut, apta_node* left, apta_node* right) = 0;

    /**
     * @brief Can be used to set the score of a refinement.
     */
    virtual double get_score(){};

    /**
     * @brief This function completes a single node with the sul.
     * 
     * An example use case would be when the train set lacks prefixes to sequences it actually contains. In the APTA those will be unlabelled states 
     * that actually do exist. Those we can complete/label with the help of an sul. 
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
     * @brief Get map of depth-of-sequence to vector of responses starting from the desired node according to the criterion set 
     * by the ii-handler. Predictions are based on the current automaton/hypothesis.
     */
    virtual layer_predictions_map predict_node_with_automaton_layer_wise(apta& aut, apta_node* node){
      throw std::invalid_argument("This ii-handler does not implement predict_node_with_automaton function");
    }
    
    /**
     * @brief Get a vector of responses starting from the desired node according to the criterion set 
     * by the ii-handler. Predictions are based on sul.
     */
    virtual std::vector<int> predict_node_with_sul(apta& aut, apta_node* node){
      throw std::invalid_argument("This ii-handler does not implement predict_node_with_automaton_layer_wise function");
    }

    /**
     * @brief Get a map of depth-of-sequence to vector of responses starting from the desired node according to the criterion set 
     * by the ii-handler. Predictions are based on sul.
     */
    virtual layer_predictions_map predict_node_with_sul_layer_wise(apta& aut, apta_node* node){
      throw std::invalid_argument("This ii-handler does not implement predict_node_with_sul_layer_wise function");
    }

    /**
     * @brief A function determining whether the distributions as gained from predict_node_with_automaton
     * and predict_node_with_sul are consistent.
     */
    virtual bool distributions_consistent(const std::vector<int>& v1, 
                                          const std::vector<int>& v2,
                                          const std::optional<int> depth1_opt = std::nullopt,
                                          const std::optional<int> depth2_opt = std::nullopt) {
      throw std::invalid_argument("This ii-handler does not implement distributions_consistent function");
    }

    #ifdef __FLEXFRINGE_CUDA
    /**
     * @brief The same as the non-CUDA version, but it does work on a GPU.
     */
    virtual bool distributions_consistent_layer_wise(const device_vector& v1,
                                                     const device_vector& v2,
                                                     const std::optional<int> depth1_opt = std::nullopt,
                                                     const std::optional<int> depth2_opt = std::nullopt) {
      throw std::invalid_argument("This ii-handler does not implement distributions_consistent function");
    }
    #else
    /**
     * @brief A function determining whether the distributions as gained from predict_node_with_automaton
     * and predict_node_with_sul are consistent. Layer-wise enables different kinds of statistical tests such as 
     * a Hoeffding-bound check, therefore we give it an extra signature.
     * 
     * Depth can be used to adjust 
     */
    virtual bool distributions_consistent_layer_wise(const layer_predictions_map& v1,
                                                     const layer_predictions_map& v2,
                                                     const std::optional<int> depth1_opt = std::nullopt,
                                                     const std::optional<int> depth2_opt = std::nullopt) {
      throw std::invalid_argument("This ii-handler does not implement distributions_consistent function");
    }
    #endif
};

/**
 * @brief Adds a suffix to the set of suffixes. Makes sure that duplicates are avoided.
 */
template<typename T> requires (std::is_same_v<T, std::vector<int>> || std::is_same_v<T, std::list<int>>)
void distinguishing_sequences_handler_base::layerwise_suffixes_t::add_suffix(const T& seq) {
  static std::unordered_set<size_t> hashed_suffixes;

  std::size_t suf_hash;
  // reinterpret_cast in following dangerous, but we only want to generate hash, therefore ok
  if constexpr(std::is_same_v< T, std::vector<int> >)
    suf_hash = get_hash(reinterpret_cast< const std::vector<uint32_t>& >(seq));
  else if constexpr(std::is_same_v< T, std::list<int> >)
    suf_hash = get_hash(reinterpret_cast< const std::list<uint32_t>& >(seq));

  if(hashed_suffixes.contains(suf_hash))
    return;

  const int length = seq.size();
  if(!m_suffixes.contains(length))
    m_suffixes[length] = std::vector< std::vector<int> >();

  hashed_suffixes.insert(suf_hash);
  if constexpr(std::is_same_v<T, std::vector<int>>){
    m_suffixes[length].push_back(seq);
  }
  else if constexpr(std::is_same_v<T, std::list<int>>){
    std::vector<int> seq_vector;
    seq_vector.reserve(seq.size());
    seq_vector.insert(seq_vector.end(), seq.begin(), seq.end());
    m_suffixes[length].push_back(std::move(seq_vector));
  }
}

#endif // __II_BASE_H__