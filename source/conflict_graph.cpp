
#include "conflict_graph.h"
using std::cout;
using std::endl;

void apta_graph::add_conflicts(state_merger* merger){
    cout << "adding conflicts " << nodes.size() << endl;

    for(merged_APTA_iterator it = merged_APTA_iterator(merger->get_aut()->get_root()); *it != nullptr; ++it) {
        graph_node *left = get_node(*it);
        merged_APTA_iterator it2 = it;
        ++it2;
        while (*it2 != nullptr) {
            graph_node *right = get_node(*it2);
            ++it2;

			if(right->anode->is_red() == true) continue;
            if(right == left) continue;

            //std::cerr << "testing: " << left->anode->get_number() << " " << right->anode->get_number() << endl;

            refinement* ref = merger->test_merge(left->anode, right->anode);
			if(ref == nullptr){
                //std::cerr << "adding: " << left->anode->get_number() << " " << right->anode->get_number() << endl;
				left->neighbors.insert(right);
				right->neighbors.insert(left);
			}
		}
	}
    print_dot();
}

apta_graph::apta_graph(state_set* states){
	for(state_set::iterator it = states->begin(); it != states->end(); ++it){
        if(*it == nullptr) continue;
	    graph_node* gn = new graph_node(*it);
		nodes.insert(gn);
		node_map[*it] = gn;
	}
	num_types = 0;
}

graph_node::graph_node(apta_node* an){
	anode = an;
}

void apta_graph::remove_edges(int size){
	node_set to_remove;
	for(node_set::iterator it = nodes.begin(); it != nodes.end(); ++it){
		if((*it)->neighbors.size() < size)
			to_remove.insert(*it);
	}
	for(node_set::iterator it = to_remove.begin(); it != to_remove.end(); ++it){
		nodes.erase(*it);
		for(node_set::iterator it2 = (*it)->neighbors.begin(); it2 != (*it)->neighbors.end(); ++it2){
			(*it2)->neighbors.erase(*it);
		}
	}
}

node_set *apta_graph::find_clique_converge() {
	int size = 0;
	while(true){
		node_set* result = find_clique();
		if(result->size() > size){
			remove_edges( result->size() );
			size = result->size();
		} else {
			return result;
		}
	}
}

void apta_graph::print_dot() {
    std::ofstream output("conflict_graph.dot");
    if (output.fail()) {
        throw std::ofstream::failure("Unable to open file for writing conflict graph");
    }
    output << "graph {\n";
    for(node_set::iterator it = nodes.begin(); it != nodes.end(); ++it) {
        graph_node* gn = *it;
        if(gn->anode == nullptr) continue;
        for(auto nit = gn->neighbors.begin(); nit != gn->neighbors.end(); nit++) {
            graph_node* an = *nit;
            if(an->anode == nullptr) continue;
            if(gn->anode->get_number() < an->anode->get_number()) output << "\t\t" << gn->anode->get_number() << " -- " << an->anode->get_number() << ";\n";
        }
    }
    output << "}\n";
}


node_set *apta_graph::find_clique(){
	node_set* result = new node_set();

	if( nodes.empty() ) return result;

	graph_node* head = NULL;
	int max_degree = -1;
	for(node_set::iterator it = nodes.begin();
			it != nodes.end();
			++it){
		graph_node* n = *it;
		if((int)n->neighbors.size() > max_degree){
			max_degree = n->neighbors.size();
			head = n;
		}
	}

	result->insert(head);

	node_set intersection = head->neighbors;

	max_degree = -1;
	for(node_set::iterator it = nodes.begin();
			it != nodes.end();
			++it){
		graph_node* n = *it;
		if((int)n->neighbors.size() > max_degree){
			max_degree = n->neighbors.size();
			head = n;
		}
	}

	while(!intersection.empty()){
		result->insert(head);
		intersection.erase(head);
		node_set new_intersection = node_set(intersection);
		for(node_set::iterator it = intersection.begin();
				it != intersection.end();
				++it){
			graph_node* keep = *it;
			if(head->neighbors.find(keep) == head->neighbors.end()){
				new_intersection.erase(keep);
			}
		}
		intersection = new_intersection;

		int best_match = -1;
		for(node_set::iterator it = intersection.begin();
				it != intersection.end();
				++it){
			graph_node* current = *it;
			int match = 0;
			for(node_set::iterator it2 = intersection.begin();
					it2 != intersection.end();
					++it2){
				if(current->neighbors.find(*it2) != current->neighbors.end()){
					match++;
				}
			}
			if(match > best_match){
				best_match = match;
				head = current;
			}
		}
	}

	return result;
}

std::pair<node_set*, node_set*> apta_graph::find_bipartite(){
    if(nodes.empty()) return std::pair<node_set*, node_set*>(NULL,NULL);

    ordered_node_set ordered_nodes = ordered_node_set(nodes);

    std::cerr << "bipartite: " << ordered_nodes.size() << endl;

	node_set* left_result = new node_set();
    node_set* right_result = new node_set();

    graph_node* left_head = *(ordered_nodes.begin());
    graph_node* right_head = NULL;
    int max_size = 0;
    for(node_set::iterator it = left_head->neighbors.begin(); it != left_head->neighbors.end(); ++it) {
        graph_node* node = *it;
        if(right_head == NULL || max_size < node->neighbors.size()){
            max_size = node->neighbors.size();
            right_head = node;
        }
    }

    std::cerr << "left head: " << left_head->anode->get_number() << "\tright head: " << right_head->anode->get_number() << endl;

    if(right_head == NULL) return std::pair<node_set*, node_set*>(left_result, right_result);

    left_result->insert(left_head);
    right_result->insert(right_head);

    while(true){
        graph_node* right_node = nullptr;
        for(node_set::iterator it = left_head->neighbors.begin(); it != left_head->neighbors.end(); ++it) {
            graph_node* node = *it;
            if(right_result->contains(node)) continue;
            //std::cerr << "\ttrying: " << node->anode->get_number() << endl;
            bool match = true;
            for(node_set::iterator it2 = left_result->begin(); it2 != left_result->end(); ++it2) {
                graph_node* right = *it2;
                if(right->neighbors.find(node) == right->neighbors.end()) {
                    match = false;
                    break;
                }
            }
            if(match){
                right_node = node;
                break;
            }
        }
        graph_node* left_node = nullptr;
        for(node_set::iterator it = right_head->neighbors.begin(); it != right_head->neighbors.end(); ++it) {
            graph_node* node = *it;
            if(left_result->contains(node)) continue;
            //std::cerr << "\ttrying: " << node->anode->get_number() << endl;
            bool match = true;
            for(node_set::iterator it2 = right_result->begin(); it2 != right_result->end(); ++it2) {
                graph_node* left = *it2;
                if(left->neighbors.find(node) == left->neighbors.end()) {
                    match = false;
                    break;
                }
            }
            if(match){
                left_node = node;
                break;
            }
        }

        if(left_node != nullptr && right_node != nullptr) {
            if (left_result->size() > right_result->size()) {
                //std::cerr << "\tadding right: " << right_node->anode->get_number() << endl;
                right_result->insert(right_node);
            } else {
                //std::cerr << "\tadding left: " << left_node->anode->get_number() << endl;
                left_result->insert(left_node);
            }
        } else if(left_node != nullptr) {
            //std::cerr << "\tadding left: " << left_node->anode->get_number() << endl;
            left_result->insert(left_node);
        } else if(right_node != nullptr) {
            //std::cerr << "\tadding right: " << right_node->anode->get_number() << endl;
            right_result->insert(right_node);
        } else {
            break;
        }
    }

    std::cerr << "bipartite subgraph: " << left_result->size() << " " << right_result->size() << endl;

    return std::pair<node_set*, node_set*>(left_result, right_result);

    for(node_set::iterator it = ordered_nodes.begin(); it != ordered_nodes.end(); ++it) {
        graph_node* node = *it;
        if(node == left_head || node == right_head) continue;

        std::cerr << "\ttrying: " << node->anode->get_number() << endl;

        // try left
        bool left_match = true;
        for(node_set::iterator it2 = right_result->begin(); it2 != right_result->end(); ++it2) {
            graph_node* right = *it2;
            if(right->neighbors.find(node) == right->neighbors.end()) {
                left_match = false;
                break;
            }
        }

        // try right
        bool right_match = true;
        for(node_set::iterator it2 = left_result->begin(); it2 != left_result->end(); ++it2) {
            graph_node* left = *it2;
            if(left->neighbors.find(node) == left->neighbors.end()) {
                right_match = false;
                break;
            }
        }

        if(left_match == true && right_match == false){
            std::cerr << "\tadding left: " << node->anode->get_number() << endl;
            left_result->insert(node);
        } else if(left_match == false && right_match == true){
            std::cerr << "\tadding right: " << node->anode->get_number() << endl;
            right_result->insert(node);
        } else if(left_match == true && right_match == true) {
            if(left_result->size() > right_result->size()){
                std::cerr << "\tadding right: " << node->anode->get_number() << endl;
                right_result->insert(node);
            } else {
                std::cerr << "\tadding left: " << node->anode->get_number() << endl;
                left_result->insert(node);
            }
        }
    }

    std::cerr << "bipartite subgraph: " << left_result->size() << " " << right_result->size() << endl;

	return std::pair<node_set*, node_set*>(left_result, right_result);
}


void apta_graph::extract_types(int min_bip_size){
    while(true){
        std::pair<node_set*, node_set*> bip = find_bipartite();
        if(bip.first == NULL || bip.second == NULL) break;
        if(bip.first->size() + bip.second->size() < min_bip_size){
            delete bip.first;
            delete bip.second;
            break;
        }
        for(node_set::iterator it = bip.first->begin(); it != bip.first->end(); ++it) {
            (*it)->pos_types.insert(num_types);
        }
        for(node_set::iterator it = bip.second->begin(); it != bip.second->end(); ++it) {
            (*it)->neg_types.insert(num_types);
        }
        for(node_set::iterator it = bip.first->begin(); it != bip.first->end(); ++it){
            for(node_set::iterator it2 = bip.second->begin(); it2 != bip.second->end(); ++it2) {
                (*it)->neighbors.erase((*it2));
                (*it2)->neighbors.erase((*it));
            }
        }
        delete bip.first;
        delete bip.second;
        num_types++;
    }
}

