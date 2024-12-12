#ifndef __LSHARP_EVAL__
#define __LSHARP_EVAL__

#include "count_types.h"

class lsharp_data: public count_data {

protected:
  REGISTER_DEC_DATATYPE(lsharp_data);

public:
};

/**
 * @brief This is a minor deviation from count_driven. It does the same, but it only checks for final distributions, 
 * making sure that the types of the nodes are consistent. Another deviation is that in the compute_score function it 
 * does merge blue nodes into lower layers if multiple merges are possible, see also the compute_score implementation.
 */
class lsharp_eval: public count_driven {

protected:
  REGISTER_DEC_TYPE(lsharp_eval);

  bool types_match(const std::unordered_map<int, int>& m1, const std::unordered_map<int, int>& m2) const noexcept;

public:
  bool consistent(state_merger *merger, apta_node* left, apta_node* right, int depth) override;
  double compute_score(state_merger* merger, apta_node* left, apta_node* right) override;
};

#endif
