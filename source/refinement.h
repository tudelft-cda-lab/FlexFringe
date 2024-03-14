
#ifndef _REFINEMENT_H_
#define _REFINEMENT_H_

using namespace std;

#include <list>
#include <queue>
#include <map>
#include <set>
#include <iostream>

// do we need all these forward-passes?
class refinement;
class merge_refinement;
class split_refinement;
class extend_refinement;
struct score_compare;
struct ref_compare;

typedef list<refinement*> refinement_list;
typedef set<refinement*, score_compare > refinement_set;
typedef set<refinement*, ref_compare > refinement_store;

class state_merger;
class apta_node;
class tail;
class trace;

enum class refinement_type{
    base_rf_type,
    merge_rf_type,
    split_rf_type,
    extend_rf_type
};

/**
 * @brief Base class for refinements. Specialized to either
 * a point (merge), split, or color (adding of a new state).
 *
 */
class refinement{
public:
    double score;
    trace* red_trace;
    apta_node* red;
    int size;
    int refs;

    int time;

    refinement();
    virtual ~refinement(){ };

    virtual void print() const;
    virtual void print_short() const;
    virtual void doref(state_merger* m);
    virtual void undo(state_merger* m);

    virtual bool testref(state_merger* m);

    virtual void increfs();

    virtual void erase();

    virtual void print_json(iostream &output) const;

    static void print_refinement_list_json(iostream &output, refinement_list *list);

    virtual refinement_type type();

    inline int get_time();

    // these two functions are for streaming
    virtual bool test_ref_structural(state_merger* m);
    virtual bool test_ref_consistency(state_merger* m);
};

/**
 * @brief A merge-refinement assigns a score to the merge of
 * a left and right state. 
 *
 */
class merge_refinement : public refinement {
public:
    trace* blue_trace;
    apta_node* blue;

    merge_refinement(state_merger* m, double s, apta_node* l, apta_node* r);
    void initialize(state_merger* m, double s, apta_node* l, apta_node* r);

    virtual inline void print() const;
    virtual inline void print_short() const;
    virtual inline void doref(state_merger* m);
    virtual inline void undo(state_merger* m);
    //virtual inline bool testref(state_merger* m);

    virtual inline void erase();

    virtual void print_json(iostream &output) const;

    virtual refinement_type type();

    // these two functions are for streaming
    virtual bool test_ref_structural(state_merger* m);
    virtual bool test_ref_consistency(state_merger* m);
};

 /**
 * @brief A extend-refinement makes a blue state red. The
 * score is the size (frequency) of the state in the APTA.
 *
 */
class extend_refinement : public refinement {
public:
	extend_refinement(state_merger* m, apta_node* r);
    void initialize(state_merger* m, apta_node* r);

	virtual inline void print() const;
	virtual inline void print_short() const;
	virtual inline void doref(state_merger* m);
	virtual inline void undo(state_merger* m);
    virtual inline bool testref(state_merger* m);

    virtual inline void erase();

    virtual void print_json(iostream &output) const;

    virtual refinement_type type();

    // these two functions are for streaming
    virtual bool test_ref_structural(state_merger* m);
    virtual bool test_ref_consistency(state_merger* m);
};

/**
 * @brief Splitting of states used for continuous values. For example, the RTI-algorithm 
 * uses this one to split states based on (continuous) time-feature.
 * 
 */
class split_refinement : public refinement {
public:
    tail* split_point;
    int attribute;

	split_refinement(state_merger* m, double s, apta_node* l, tail* t, int a);
	void initialize(state_merger* m, double s, apta_node* l, tail* t, int a);

	virtual inline void print() const;
	virtual inline void print_short() const;
	virtual inline void doref(state_merger* m);
	virtual inline void undo(state_merger* m);
    virtual inline bool testref(state_merger* m);

    virtual inline void erase();

    virtual void print_json(iostream &output) const;

    virtual refinement_type type();

    // these two functions are for streaming
    virtual bool test_ref_structural(state_merger* m);
    virtual bool test_ref_consistency(state_merger* m);
};

 /**
 * @brief Compare function for refinements, based on scores.
 *
 */
struct score_compare {
    inline bool operator()(refinement* r1, refinement* r2) const {
        //if(r1->type() != r2->type()) return r1->type() < r2->type();
        if(r1->score == r2->score) return r1->size > r2->size;
        return r1->score > r2->score;
    }
};

/**
* @brief Compare function for refinements, based on scores.
*
*/
struct ref_compare {
    inline bool operator()(refinement* r1, refinement* r2) const {
        if(r1->type() != r2->type()){
            if(r1->type()==refinement_type::split_rf_type || (r1->type()==refinement_type::merge_rf_type && r2->type()==refinement_type::extend_rf_type)){
                return true;
            }
            return false;
        }
        if(r1->type() == refinement_type::split_rf_type){
            split_refinement* sr1 = (split_refinement*)r1;
            split_refinement* sr2 = (split_refinement*)r2;
            return sr1->red < sr2->red;
        }
        if(r1->type() == refinement_type::merge_rf_type){
            merge_refinement* mr1 = (merge_refinement*)r1;
            merge_refinement* mr2 = (merge_refinement*)r2;
            if(mr1->red != mr2->red) return mr1->red < mr2->red;
            return mr1->blue < mr2->blue;
        }
        if(r1->type() == refinement_type::extend_rf_type){
            extend_refinement* er1 = (extend_refinement*)r1;
            extend_refinement* er2 = (extend_refinement*)r2;
            return er1->red < er2->red;
        }
        return r1->score > r2->score;
    }
};

#endif /* _REFINEMENT_H_ */
