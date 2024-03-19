#ifndef __MEMSTORE_H__
#define __MEMSTORE_H__

#include <list>
#include "refinement.h"
#include "apta.h"
#include "state_merger.h"

#include "input/inputdata.h"
#include "input/trace.h"
#include "input/tail.h"

class mem_store {
private:
    static std::list<trace*> trace_store;
    static std::list<tail*> tail_store;
public:
    static std::list< apta_node* > node_store;
    static std::list< apta_guard* > guard_store;
//    static std::list< tail* > tail_store;
//    static std::list< trace* > trace_store;
    static std::list< merge_refinement* > mergeref_store;
    static std::list< split_refinement* > splitref_store;
    static std::list< extend_refinement* > extendref_store;

    static void delete_node(apta_node*);
    static apta_node* create_node(apta_node* other_node);

    static void delete_guard(apta_guard*);
    static apta_guard* create_guard(apta_guard* other_guard);

//    static void delete_tail(tail*);
//    static tail* create_tail(tail* other_tail);

    static void delete_merge_refinement(merge_refinement*);
    static merge_refinement* create_merge_refinement(state_merger* m, double s, apta_node* l, apta_node* r);

    static void delete_split_refinement(split_refinement*);
    static split_refinement* create_split_refinement(state_merger* m, double s, apta_node* l, tail* t, int a);

    static void delete_extend_refinement(extend_refinement*);
    static extend_refinement* create_extend_refinement(state_merger* m, apta_node* r);

    static void erase();

//    static void delete_trace(trace*);
//    static trace* create_trace();

    static void delete_trace(trace*);
    static trace* create_trace(inputdata* = nullptr);

    static void delete_tail(tail*);
    static tail* create_tail(tail* other_tail);
};

#endif
