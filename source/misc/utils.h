// -------------------------------------------------------------------
// --- Reversed iterable https://stackoverflow.com/a/28139075/8477066

#include <iterator>

namespace utils {

template <typename T> struct reversion_wrapper { T& iterable; };

template <typename T> auto begin(reversion_wrapper<T> w) { return std::rbegin(w.iterable); }

template <typename T> auto end(reversion_wrapper<T> w) { return std::rend(w.iterable); }

template <typename T> reversion_wrapper<T> reverse(T&& iterable) { return {iterable}; }

// There is a very new C++20 feature std::views::keys that only works with clang 16 or newer.
// Sicco uses 15 last time I checked (Jan 2024)
// So in the meantime you can use this from:
// https://stackoverflow.com/a/42534128/8477066
// Update: I discovered Clang on Linux is different from Clang on MacOS, so maybe he has support.

template <class MapType> class MapKeyIterator {
  public:
    class iterator {
      public:
        iterator(typename MapType::iterator it) : it(it) {}
        iterator operator++() { return ++it; }
        bool operator!=(const iterator& other) { return it != other.it; }
        typename MapType::key_type operator*() const { return it->first; } // Return key part of map
      private:
        typename MapType::iterator it;
    };

  private:
    MapType& map;

  public:
    MapKeyIterator(MapType& m) : map(m) {}
    iterator begin() { return iterator(map.begin()); }
    iterator end() { return iterator(map.end()); }
};
template <class MapType> MapKeyIterator<MapType> map_keys(MapType& m) { return MapKeyIterator<MapType>(m); }

} // namespace utils
