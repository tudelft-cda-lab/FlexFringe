#ifndef __APTA_H__
#define __APTA_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>
#include <map>
#include <unordered_map>
#include <string>
#include <queue>

#include "input/trace.h"

class apta;
class apta_node;
class APTA_iterator;
struct size_compare;
class evaluation_data;
class apta_guard;

typedef std::list<int> int_list;
typedef std::list<double> double_list;

typedef std::pair<bool, double> score_pair;
typedef std::pair< std::pair<int, int>, score_pair > ts_pair;
typedef std::map< apta_node*, ts_pair > score_map;
typedef std::list< std::pair< int, int > > size_list;

typedef std::set<apta_node*, size_compare> state_set;

typedef std::list<int> int_list;
typedef std::list<double> double_list;

typedef std::map<int, int> num_map;

typedef std::multimap<int, apta_guard*> guard_map;
typedef std::map<int, double> bound_map;

#include "parameters.h"
#include "evaluate.h"
#include "input/inputdata.h"

typedef std::list< std::pair< tail*, int > > split_list;

class apta_guard{
private:
    /** the target of the transition guard */
    apta_node* target;
    /** the guard bounds (attribute, value) pairs */
    bound_map min_attribute_values;
    bound_map max_attribute_values;

    /** due to splitting, the target may change (for speedup)
     * this stores the original target to undo the operation */
    apta_node* undo_split;

public:
    inline apta_node* get_target(){ return target; }

    /** constructors and initializer */
    apta_guard();
    apta_guard(apta_guard*);
    void initialize(apta_guard *g);

    /** returns true if the attribute values in t satisfy the guard bounds */
    bool bounds_satisfy(tail* t);

    friend class apta_node;
    friend class apta;
    friend class APTA_iterator;
    friend class merged_APTA_iterator;
    friend class merged_APTA_iterator_func;
    friend class blue_state_iterator;
    friend class red_state_iterator;
    friend class tail_iterator;
    friend class inputdata;
    friend class state_merger;
};

/** iterators for the APTA and merged APTA nodes, performs breadth-first traversal
 * invariants:
 * - the apta is a tree, following sources leads to the root
 * - every apta_node has a source
 * - the red states may contain loops/recursion
 * - every blue state is the root of a tree
 * - nodes in such a tree can be merged, but the result must be a tree */
class APTA_iterator {
public:
    apta_node* base;
    apta_node* current;

    std::queue<apta_node*> q;
    
    APTA_iterator(apta_node* start);

    virtual void increment();
    
    apta_node* operator*() const { return current; }
    APTA_iterator& operator++() { increment(); return *this; }
};

class merged_APTA_iterator {
public:
    apta_node* base;
    apta_node* current;

    std::queue<apta_node*> q;

    merged_APTA_iterator(apta_node* start);

    virtual void increment();
    
    apta_node* operator*() const { return current; }
    merged_APTA_iterator& operator++() { increment(); return *this; }
};

class blue_state_iterator : public merged_APTA_iterator {
public:
    
    blue_state_iterator(apta_node* start);

    virtual void increment();
};

class red_state_iterator : public merged_APTA_iterator {
public:
    
    red_state_iterator(apta_node* start);

    virtual void increment();
};

class merged_APTA_iterator_func : public merged_APTA_iterator {
public:
    
    bool(*check_function)(apta_node*);

    merged_APTA_iterator_func(apta_node* start, bool(*)(apta_node*));

    virtual void increment();
};

class tail_iterator {
public:
    apta_node* base;
    apta_node* current;
    tail* current_tail;
    
    tail_iterator(apta_node* start);
    
    apta_node* next_forward();
    virtual void increment();
    void next_node();
    
    tail* operator*() const { return current_tail; }
    tail_iterator& operator++() { increment(); return *this; }
};

typedef std::set<apta_node*, size_compare> state_set;

#include "evaluate.h"

/** 
 * @brief Data structure for the  prefix tree.
 *
 * The prefix tree is a pointer structure made
 * of apta_nodes.
 * @see apta_node
 */

class apta{
private:
    state_merger* merger; /**< merger context for convenience */
    apta_node* root; /**< root of the tree */

public:
    inline apta_node* get_root(){ return root; }
    inline state_merger* get_context(){ return merger; }

    apta();
    ~apta();
    inline void set_context(state_merger* m){ merger = m; }

    /** reading and writing an apta to and from file */
    void print_dot(std::iostream& output);
    void print_json(std::iostream& output);
    void read_json(std::istream &input_stream);
    void print_sinks_json(std::iostream &output);

    /** for better layout when visualizing state machines from the json file
     * set nodes to this depth in a hierarchical view */
    void set_json_depths();

    friend class apta_guard;
    friend class APTA_iterator;
    friend class merged_APTA_iterator;
    friend class merged_APTA_iterator_func;
    friend class blue_state_iterator;
    friend class red_state_iterator;
    friend class tail_iterator;
    friend class inputdata;
    friend class IInputData; // TODO: rename
    friend class state_merger;

    bool print_node(apta_node *n);
};

/**
 * @brief Node structure in the prefix tree.
 *
 * The prefix tree is a pointer structure made
 * of apta_nodes.
 */
class apta_node{
private:
    /** access trace for reaching a state from the root
     * depending on parameters, either stores entire trace or last tail */
    trace* access_trace;
    /** parent state in the prefix tree */
    apta_node* source;
    /** is this a red state? */
    bool red;
    /** is this a sink state? denotes sink type */
    int sink;
    /** depth of the node in the apta */
    int depth;
    /** unique state identifiers, used by SAT encoding and reading/writing */
    int number;

    /** transitions to child states */
    guard_map guards;
    /** UNION/FIND data structure for merges */
    apta_node* representative;
    /** storing all states merged with this state
     * to get access to all tails and incoming transitions */
    apta_node* next_merged_node;
    apta_node* representative_of;
    /** UNION/FIND size measure, number of state occurrences
     * size = total count, final = ending occurrences */
    int size;
    int final;

    /** merge score, stored after performing a merge */
    double merge_score;

    /** variables used for splitting */
    /** singly linked list containing all tails in this state */
    tail* tails_head;
    void add_tail(tail* t);
    /** list of previously performed splits in this state, stored for pre_splitting */
    split_list* performed_splits;
    /** the source can change due to splitting (we do not create new nodes when all tails are split)
     * this stores the original to undo_merge splits */
    apta_node* original_source;

    /** extra information for merging heuristics and consistency checks
     * gets overloaded with evaluation functions such as Alergia, EDSM, ... */
    evaluation_data* data;

public:
    inline trace* get_access_trace(){ return access_trace; }
    inline apta_node* get_source(){ return source; }
    inline apta_node* get_merged_head(){ return representative_of; }
    inline apta_node* get_next_merged(){ return next_merged_node; }
    inline evaluation_data* get_data(){ return data; }
    inline int get_number(){ return number; }
    inline int get_size(){ return size; }
    inline int get_final(){ return final; }
    inline int get_depth(){ return depth; }
    inline double get_score(){ return merge_score; }
    inline void set_score(double m){  merge_score = m; }
    inline void set_red(bool b){ red = b; };
    inline apta_node* rep(){ return representative; }

    /** this gets merged with node, replacing head of list */
    inline void merge_with(apta_node* node){
        assert(this->representative == nullptr);
        this->representative = node;
        this->next_merged_node = node->representative_of;
        node->representative_of = this;

        node->size += this->size;
        node->final += this->final;
    };
    /** undo_merge this gets merged with node, resetting head of list */
    inline void undo_merge_with(apta_node* node){
        assert(this->representative == node);
        this->representative = 0;
        node->representative_of = this->next_merged_node;
        this->next_merged_node = 0;

        node->size -= this->size;
        node->final -= this->final;
    };

    /** FIND/UNION functions, returns head of representative list */
    inline apta_node* find(){
        apta_node* rep = this;
        while(rep->representative != 0) rep = rep->representative;
        return rep;
    };
    /** FIND/UNION functions, returns rep that has node as representative */
    inline apta_node* find_until(apta_node* node){
        apta_node* rep = this;
        while(rep->representative != 0 && rep->representative != node){
            rep = rep->representative;
        }
        return rep;
    };

    /** getting target states with bounded guards, access via tails or guard from other state */
    apta_node* child(tail* t);
    apta_guard* guard(int i, apta_guard* g);
    apta_guard* guard(tail* t);
    void set_child(tail* t, apta_node* node);

    /** getting target states via symbols only */
    inline apta_node* child(int i){
        guard_map::iterator it = guards.find(i);
        if(it != guards.end()) return it->second->target;
        return 0;
    };
    inline apta_node* merged_child(int i){
        guard_map::iterator it = guards.find(i);
        if(it != guards.end()) return it->second->target->find();
        return 0;
    };
    inline apta_guard* guard(int i){
        guard_map::iterator it = guards.find(i);
        if(it != guards.end()) return it->second;
        return 0;
    };
    inline guard_map::iterator guards_start(){ return guards.begin(); }
    inline guard_map::iterator guards_end(){ return guards.end(); }
    inline void set_child(int i, apta_node* node){
        guard_map::iterator it = guards.find(i);
        if(it != guards.end()){
            if(node != 0)
                it->second->target = node;
            else
                guards.erase(it);
        } else {
            apta_guard* g = new apta_guard();
            guards.insert(std::pair<int,apta_guard*>(i,g));
            g->target = node;
        }
    };
    inline apta_node* get_child(int c){
        apta_node* rep = find();
        if(rep->child(c) != 0) return rep->child(c)->find();
        return 0;
    };

    /** red, blue, white, and sinks */

    inline bool is_red() const{
        return red;
    }
    inline bool is_blue() const{
        return source != 0 && is_red() == false && source->find()->is_red();
    }
    inline bool is_white() const{
        return source != 0 && is_red() == false && !source->find()->is_red();
    }
    inline bool is_sink() const{
        if(sink != -1) return true;
        return data->sink_type() != -1;
    }
    inline int sink_type() const{
        if(sink != -1) return sink;
        return data->sink_type();
    }

    /** constructors and intializers */
    apta_node();
    ~apta_node();
    void initialize(apta_node* n);

    /** print to json output, use later in predict functions */
    void print_json(json &output);
    void print_json_transitions(json &output);
    void print_dot(std::iostream& output);

    /** below are functions use by special heuristics/settings and output printing */
    friend class apta;
    friend class apta_guard;
    friend class APTA_iterator;
    friend class merged_APTA_iterator;
    friend class merged_APTA_iterator_func;
    friend class blue_state_iterator;
    friend class red_state_iterator;
    friend class tail_iterator;
    friend class inputdata;
    friend class IInputData; // TODO: rename
    friend class state_merger;

    std::set<apta_node *> *get_sources();
};

struct size_compare
{
    bool operator()(apta_node* left, apta_node* right) const
    {
        if(DEPTH_FIRST){
            if(left->get_depth() > right->get_depth())
                return 1;
            if(left->get_depth() < right->get_depth())
                return 0;
        }
        if(left->get_size() > right->get_size())
            return 1;
        if(left->get_size() < right->get_size())
            return 0;
        return left->get_number() < right->get_number();
    }
};

#endif
