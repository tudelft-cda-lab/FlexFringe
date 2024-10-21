/**
 * @file types_state_comparator.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Compares the type and a hidden representation of two nodes. We currently use this 
 * one in active learning, when querying transformers.
 * 
 * TODO: can we find a better structure for these here? It is a simpler derivative of the 
 * count_types fused with the weight state comparator.
 * 
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __TYPE_STATE_COMPARATOR_H__
#define __TYPE_STATE_COMPARATOR_H__

#include "count_types.h"

#include <vector>

/* The data contained in every node of the prefix tree or DFA */
class type_state_comparator_data: public evaluation_data {

protected:
  REGISTER_DEC_DATATYPE(type_state_comparator_data);
  std::vector<float> hidden_state;
  int node_type;

public:
    num_map final_counts;
    num_map path_counts;
    
    int total_final;
    int total_paths;

    type_state_comparator_data();
    
    virtual void print_transition_label(std::iostream& output, int symbol);
    virtual void print_state_label(std::iostream& output);

    virtual void print_transition_label_json(std::iostream& output, int symbol);
    virtual void print_state_label_json(std::iostream& output);

    virtual void update(evaluation_data* right);
    virtual void undo(evaluation_data* right);

    void initialize_state(const std::vector<float>& state){
        this->hidden_state = state;
    }
    
    inline int num_paths() const noexcept {
        return total_paths;
    }
    
    inline int num_final() const noexcept {
        return total_final;
    }

    inline int num_total() const noexcept {
        return total_final + total_paths;
    }

    inline int get_type() const noexcept { 
      return node_type; 
    }

    inline const std::vector<float>& get_state() const noexcept { 
      return hidden_state;  
    }

    virtual void initialize();

    virtual void del_tail(tail *t);
    virtual void add_tail(tail *t);

    virtual void read_json(json& node);
    virtual void write_json(json& node);

    virtual int predict_type(tail*);

    //int get_type_sink();
    //bool is_low_count_sink();
    //virtual int sink_type();
    //virtual bool sink_consistent(int type);
    //virtual int num_sink_types();
};

class type_state_comparator: public evaluation_function {

protected:
  REGISTER_DEC_TYPE(type_state_comparator);

public:
  int num_merges;

  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual double compute_score(state_merger*, apta_node* left, apta_node* right);
  virtual void reset(state_merger *merger);
  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right, int depth);

  double compute_state_distance(apta_node* left_node, apta_node* right_node);
};

#endif
