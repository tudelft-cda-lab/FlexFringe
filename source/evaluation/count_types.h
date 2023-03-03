#ifndef __COUNT__
#define __COUNT__

#include "evaluate.h"

typedef map<int, int> num_map;

/* The data contained in every node of the prefix tree or DFA */
class count_data: public evaluation_data {

protected:
  REGISTER_DEC_DATATYPE(count_data);

public:
    num_map final_counts;
    num_map path_counts;
    
    int total_final;
    int total_paths;

    count_data();
    
    virtual void print_transition_label(iostream& output, int symbol);
    virtual void print_state_label(iostream& output);

    virtual void print_transition_label_json(iostream& output, int symbol);
    virtual void print_state_label_json(iostream& output);

    virtual void update(evaluation_data* right);
    virtual void undo(evaluation_data* right);

    virtual int sink_type();
    virtual bool sink_consistent(int type);
    virtual int num_sink_types();
    
    inline int num_paths(){
        return total_paths;
    }
    
    inline int num_final(){
        return total_final;
    }

    inline int num_total(){
        return total_final + total_paths;
    }

    inline int num_paths(int type){
        return path_counts[type];
	    //return path_counts.at(type);
    }

    inline int num_final(int type){
        return final_counts[type];
    }

    inline int num_total(int type){
        return path_counts[type] + final_counts[type];
    }

    inline int pos_paths(){
        return path_counts[1];
    }

    inline int neg_paths(){
        return path_counts[0];
    }

    inline int pos_final(){
        return final_counts[1];
    }

    inline int neg_final(){
        return final_counts[0];
    }

    inline int pos_total(){
        return path_counts[1] + final_counts[1];
    }

    inline int neg_total(){
        return path_counts[0] + final_counts[0];
    }

    inline int get_final_type(){
        for(auto val : final_counts){
            if(val.second == num_final()) return val.first;
        }
        return -1;
    }
    inline int get_path_type(){
        for(auto val : path_counts){
            if(val.second == num_final()) return val.first;
        }
        return -1;
    }

    virtual void initialize();

    virtual void del_tail(tail *t);

    virtual void add_tail(tail *t);

    virtual void read_json(json& node);
    virtual void write_json(json& node);

    virtual int predict_type(tail*);
    virtual int predict_path_type(tail*);

    virtual double predict_type_score(int t);
    virtual double predict_path_type_score(int t);

    int get_type_sink();
    bool is_low_count_sink();
};

class count_driven: public evaluation_function {

protected:
  REGISTER_DEC_TYPE(count_driven);

public:
  int num_merges;

  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);
  virtual void reset(state_merger *merger);
  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right);
};

#endif
