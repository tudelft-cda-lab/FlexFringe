
#ifndef _CONFLICT_GRAPH_H_
#define _CONFLICT_GRAPH_H_

#include <set>
#include "state_merger.h"
#include "apta.h"

class apta_graph;
class graph_node;

typedef std::set<graph_node*> node_set;
typedef std::set<int> type_set;

class graph_node{
public:
	node_set neighbors;
	apta_node* anode;

    type_set pos_types;
    type_set neg_types;
    inline bool type_consistent(graph_node* other){
        type_set::iterator it = pos_types.begin();
        type_set::iterator it2 = other->neg_types.begin();
        while(it != pos_types.end() && it2 != other->neg_types.end()){
            int pos_type = *it;
            int neg_type = *it2;
            if(pos_type == neg_type) return false;
            else if(pos_type < neg_type) ++it;
            else ++it2;
        }
        return true;
    }

	graph_node(apta_node*);
};

struct neighbor_compare
{
    bool operator()(graph_node* left, graph_node* right) const
    {
        if(left->neighbors.size() > right->neighbors.size())
            return 1;
        if(left->neighbors.size() < right->neighbors.size())
            return 0;
        return left < right;
    }
};

typedef std::set<graph_node*, neighbor_compare> ordered_node_set;

class apta_graph{
public:
	ordered_node_set nodes;
	std::map<apta_node*, graph_node*> node_map;
	inline graph_node* get_node(apta_node* n){
	    return node_map[n];
	}

	int num_types;

	apta_graph(state_set* states);
  
	node_set* find_clique();
	node_set* find_clique_converge();
	void remove_edges(int);
	void add_conflicts(state_merger* merger);

    std::pair<node_set*, node_set*> find_bipartite();
    void extract_types(int min_bip_size);
};


#endif /* _CONFLICT_GRAPH_H_ */
