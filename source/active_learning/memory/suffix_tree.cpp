/**
 * @file suffix_tree.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "suffix_tree.h"

#include <stack>
#include <unordered_set>

using namespace std;

/**
 * @brief Adds the suffix to the tree.
 * 
 * @param suffix The suffix.
 */
void suffix_tree::add_suffix(const list<int>& suffix){
  sf_tree_node* node = root;

  for(auto symbol: suffix){
    if(!node->has_child(symbol)){
      node_store.push_back(sf_tree_node());
      sf_tree_node& child = node_store.back();

      node->add_child(symbol, child);
    }

    node = node->get_child(symbol);
  }

  if(!node->is_final()){
    ++n_final;
    node->finalize();
  }
}

bool suffix_tree::contains(const list<int>& suffix){
  sf_tree_node* node = root;

  for(auto symbol: suffix){
    if(node->has_child(symbol))
      sf_tree_node* child = node->get_child(symbol);
    else
      return false;
  }

  return node->is_final();
}

/**
 * @brief We do a DFS search* here , looking for the next sequence to return. If the
 * search is exhausted, then the function returns a nullopt and resets to the beginning.
 * 
 * *(DFS because like this we can maintain the suffix string better)
 * 
 * @return optional< vector<int> > The 
 */
optional< vector<int> > suffix_tree::next() const noexcept {
  static vector<int> search_suffix;
  static sf_tree_node* current_node = root;
  static stack< pair<sf_tree_node*, int> > search_stack;
  static unordered_set<sf_tree_node*> seen_nodes;

  // note: the empty string is not considered a distinguishing sequence
  if(current_node==root){
    for(auto& n_s_pair: current_node->get_children()){
      search_stack.push(make_pair(n_s_pair.first, n_s_pair.second));
    }
    seen_nodes.insert(root);
  }

  while(!search_stack.empty()){
    pair<sf_tree_node*, int> curr_pair = move(search_stack.top()); // TODO: can this move-cast lead to problems?
    search_stack.pop();
    
    sf_tree_node* node = curr_pair.first;
    search_suffix.push_back(curr_pair.second);
    seen_nodes.insert(node);

    auto children = node->get_children();
    for(auto& n_s_pair: children){
      if(!seen_nodes.contains(n_s_pair.first))
        search_stack.emplace(n_s_pair.first, n_s_pair.second);
    }

    if(node->is_final()){
      return make_optional(search_suffix);
    }

    if(children.empty())
      search_suffix.pop_back();
  }

  // reset the search to the beginning
  current_node = root;
  search_suffix.clear();
  return nullopt;
}


/**
 * @brief Sets child as a child node with incoming symbol s to this node.
 * 
 * @param s Symbol leading to child node.
 * @param child The child node we want to add.
 */
void sf_tree_node::add_child(const int s, sf_tree_node& child) noexcept {
  if(symbol_child_map.contains(s))
    return;
  symbol_child_map[s] = &child;
}

/**
 * @brief Get the child object as a pointer.
 * 
 * Important: Does not check if child exists in the map of children. Make sure that child exists before calling.
 * 
 */
sf_tree_node* sf_tree_node::get_child(const int s) noexcept {
  return symbol_child_map[s];
}

/**
 * @brief Get all children of node. Helps iterating.
 */
vector< pair<sf_tree_node*, int> > sf_tree_node::get_children() noexcept {
  vector<pair<sf_tree_node*, int> > res;
  for(const auto& [symbol, child_node_ptr] : symbol_child_map){
    res.push_back(make_pair(child_node_ptr, symbol));
  }
  return res;
}
