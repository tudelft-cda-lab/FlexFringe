#include <iostream>
#include <vector>
#include <map>
#include <set>

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
  o << "[";
  if (v.empty()) {
    o << "]";
    return o;
  }
  // For every item except the last write "Item, "
  for (auto it = v.begin(); it != --v.end(); it++) {
    o << *it << ", ";
  }
  // Write out the last item
  o << v.back() << "]";
  return o;
}

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::set<T>& v) {
  o << "{";
  if (v.size() == 1) {
    o << "}";
    return o;
  }
  // For every item except the last write "Item, "
  for (auto it : v ) {
    o << it; 
    if (it != *v.rbegin()) o << ", ";
  }
  // Write out the last item
  o << "}";
  return o;
}

template <typename T, typename S> 
std::ostream& operator<<(std::ostream& os, const std::pair<T, S>& v) { 
    os << "("; 
    os << v.first << ", " << v.second << ")"; 
    return os; 
} 
  
template <typename KeyT, typename ValueT>
std::ostream& operator<<(std::ostream& o, const std::map<KeyT, ValueT>& m) {
    o << "{";
    if (m.empty()) {
        o << "}";
        return o;
    }
    // For every pair except the last write "Key: Value, "
    for (auto it = m.begin(); it != --m.end(); it++) {
        const auto& [key, value] = *it;
        o << key << ": " << value << ", ";
    }
    // Write out the last item
    const auto& [key, value] = *--m.end();
    o << key << ": " << value << "}";
    return o;
}
