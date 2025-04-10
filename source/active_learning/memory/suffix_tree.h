/**
 * @file suffix_tree.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The tree representing a set of suffixes. A simple data structure that saves us space, and only capable of representing the 
 * existence of a suffix/sequence.
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _SUFFIX_TREE_H_
#define _SUFFIX_TREE_H_

#include <vector>
#include <list>
#include <unordered_map>
#include <stack>
#include <optional>

#include <utility>
#include <concepts>
#include <stdexcept>

struct sf_tree_node {
  private:
    bool terminated = false; // used to make sure that this is the final sequence
    std::unordered_map<int, sf_tree_node*> symbol_child_map;

    const int DEPTH; // keeping track of the depth helps us writing the DFS in iterative manner (answers: how many suffixes to pop back?)
    
  public:
    sf_tree_node(const int depth) : DEPTH(depth){};

    void add_child(const int s, sf_tree_node& child) noexcept;
    inline sf_tree_node* get_child(const int s) noexcept;
    bool has_child(const int s) noexcept {return symbol_child_map.contains(s);}

    std::vector< std::pair<sf_tree_node*, int> > get_children() noexcept;

    void finalize() noexcept {terminated = true;}
    bool is_final() const noexcept {return terminated;}
    int get_depth() const noexcept {return DEPTH;}
};

class suffix_tree {
  private:
    // those are static to enable multithreading. Using static members rather than pointer structures used less coding time
    inline static sf_tree_node* root;
    inline static std::list<sf_tree_node> node_store; // making sure we do not lose them
    inline static int n_final = 0;

    // the following block represents the status of the current search
    std::vector<int> search_suffix;
    sf_tree_node* current_node;
    std::stack< std::pair<sf_tree_node*, int> > search_stack;
    int last_depth = 0;

  public:
    template<class T> requires (std::is_same_v<T, std::list<int>> || std::is_same_v<T, std::vector<int>>)
    bool add_suffix(const T& suffix);

    template<class T> requires (std::is_same_v<T, std::list<int>> || std::is_same_v<T, std::vector<int>>)
    bool contains(const T& suffix);

    std::optional< std::vector<int> > next() noexcept;
    int size() const noexcept { return n_final; }

    suffix_tree(){
      if(node_store.size()==0){
        node_store.push_back(sf_tree_node(0));
        root = &(node_store.back());
      }
      current_node = root;
    }

    suffix_tree(const suffix_tree& other) = delete;
    suffix_tree(suffix_tree&& other) = delete;
};

/**
 * @brief Adds the suffix to the tree.
 * 
 * @param suffix The suffix.
 * @return True if suffix has not already been added before (if number of suffixes increased by one), else false. 
 */
template<typename T> requires (std::is_same_v<T, std::list<int>> || std::is_same_v<T, std::vector<int>>)
bool suffix_tree::add_suffix(const T& suffix){
  sf_tree_node* node = root;

  int depth = 0;
  for(auto symbol: suffix){
    ++depth;

    if(!node->has_child(symbol)){
      node_store.push_back(sf_tree_node(depth));
      sf_tree_node& child = node_store.back();

      node->add_child(symbol, child);
    }
    node = node->get_child(symbol);
  }

  if(!node->is_final()){
    ++n_final;
    node->finalize();
    return true;
  }

  return false;
}

template<typename T> requires (std::is_same_v<T, std::list<int>> || std::is_same_v<T, std::vector<int>>)
bool suffix_tree::contains(const T& suffix){
  sf_tree_node* node = root;

  for(auto symbol: suffix){
    if(node->has_child(symbol))
      sf_tree_node* child = node->get_child(symbol);
    else
      return false;
  }

  return node->is_final();
}

#endif