#ifndef _EVALUATE_H_
#define _EVALUATE_H_

#include <vector>
#include <set>
#include <list>
#include <map>
#include <sstream>
#include "json.hpp"

class evaluation_data;
class evaluation_function;
class apta_node;
class tail;

#include "evaluation_factory.h"
#include "utility/loguru.hpp"
#include "state_merger.h"

using json = nlohmann::json;

bool is_stream_sink(apta_node*);

// for registering evaluation data objects
#define REGISTER_DEC_DATATYPE(NAME) \
    static DerivedDataRegister<NAME> reg

#define REGISTER_DEF_DATATYPE(NAME) \
    DerivedDataRegister<NAME> NAME::reg(#NAME)


// for registering evaluation function objects
#define REGISTER_DEC_TYPE(NAME) \
    static DerivedRegister<NAME> reg

#define REGISTER_DEF_TYPE(NAME) \
    DerivedRegister<NAME> NAME::reg(#NAME)


// for auto-generating names
inline std::string className(const std::string& prettyFunction)
{
    size_t colons = prettyFunction.find("::");
    if (colons == std::string::npos)
        return "::";
    size_t begin = prettyFunction.substr(0,colons).rfind(" ") + 1;
    size_t end = colons - begin;

    return prettyFunction.substr(begin,end);
}

#define __CLASS_NAME__ className(__PRETTY_FUNCTION__)

/**
 * @brief Extra data attached to each symbol, stored in each node.
 *
 * Stores all the extra data attached to symbols in a node. The 
 * class is used by the evaluation_function class.
 * It also provides the sink functions determining when to ignore
 * a node due to user-defined criteria, i.e. low frquency counts.
 * Important: Each evaluation_data type has to be registered by
 * calling REGISTER_DEF_DATATYPE(<unique name>).
 * @see evaluation_function
 */
class evaluation_data {

protected:
    static DerivedDataRegister<evaluation_data> reg;

public:
    apta_node* node;

    int node_type;
    evaluation_data* undo_pointer;

    bool undo_consistent;
    double undo_score;

    evaluation_data();
    virtual ~evaluation_data(){ };

/** Set values from input string */
    virtual void add_tail(tail* t);
    virtual void del_tail(tail* t);

/** Update values when merging */
    virtual void update(evaluation_data* other);
/** Undo updates when undoing merge */
    virtual void undo(evaluation_data* other);
/** Update values when splitting */
    virtual void split_update(evaluation_data* other);
/** Undo updates when undoing a split */
    virtual void split_undo(evaluation_data* other);

/** Printing of nodes and transitions in dot output */
    virtual void print_state_label(std::iostream& output);
    virtual void print_state_style(std::iostream& output);
    virtual void print_transition_label(std::iostream& output, int symbol);
    virtual void print_transition_style(std::iostream& output, std::set<int> symbols);

/** Printing of nodes and transitions in json output */
    virtual void print_state_label_json(std::iostream& output);
    virtual void print_transition_label_json(std::iostream& output, int symbol);

/** Print state/transition properties  */
    virtual void print_state_properties(std::iostream& output);
    virtual void print_transition_properties(std::iostream&, int symbol);

/** what to ignore, and why */
    
    virtual bool sink_consistent(int type);

/** Return a sink type, or -1 if no sink
 * Sinks are special states that optionally are not considered as merge candidates,
 * and are optionally merged into one (for every type) before starting exact solving */
    virtual int sink_type();
    virtual bool sink_consistent(apta_node* node, int type);
    virtual int num_sink_types();

    virtual void initialize();

    virtual bool print_state_true();

    virtual void read_json(json& node);
    virtual void write_json(json& node);

    virtual int predict_type(tail*);
    virtual int predict_symbol(tail*);
    virtual double predict_attr(tail*, int attr);
    virtual std::string predict_data(tail*);

    virtual double predict_score(tail* t);
    virtual double predict_type_score(int t);
    virtual double predict_type_score(tail* t);
    virtual double predict_symbol_score(int s);
    virtual double predict_symbol_score(tail* t);
    virtual double predict_attr_score(int attr, double v);
    virtual double predict_attr_score(int attr, tail* t);
    virtual double predict_data_score(std::string s);
    virtual double predict_data_score(tail* t);

    virtual tail* sample_tail();

    virtual double align_score(tail* t);

    void set_context(apta_node *n);

    virtual void print_state_style_json(std::iostream &output);

    bool align_consistent(tail *t);
};


/**
 * @brief User-defined merge heuristic function.
 *
 * Important: Each evaluation_data type has to be registered by
 * calling REGISTER_DEF_TYPE(<unique name>).
 * @see evaluation_data
 */
class evaluation_function  {

protected:
    static DerivedRegister<evaluation_function> reg;
    std::string evalpar;

    state_merger* merger;

public:

/** Constructors */
    evaluation_function();
    virtual ~evaluation_function(){ };

/** Global data */
    bool inconsistency_found;
    int num_inconsistencies;
    int num_merges;

    void set_params(std::string params);

/** Boolean indicating the evaluation function type;
   are are two kinds: computed before or after/during a merge.
   When computed before a merge, a merge is only tried for consistency.
   Functions computed before merging (typically) do not take loops that
   the merge creates into account.
   Functions computed after/during a merge rely heavily on the determinization
   process for computation, this is a strong assumption. */
    bool compute_before_merge;

/** A set containing the left states that have been merged already
   some evaluation methods use it for making different calculations */
    std::set<apta_node*> merged_left_states;
    inline bool already_merged(apta_node* left){
        return merged_left_states.find(left) != merged_left_states.end();
    };


/** An evaluation function needs to implement all of these functions */

/** Called when performing a merge, for every pair of merged nodes,
* compute the local consistency of a merge and update stored data values
*
* huge influence on performance, needs to be simple */
    virtual bool consistent(state_merger*, apta_node* left, apta_node* right);
    virtual void update_score(state_merger*, apta_node* left, apta_node* right);
    virtual void update_score_after(state_merger*, apta_node* left, apta_node* right);

    virtual bool split_consistent(state_merger*, apta_node* left, apta_node* right);
    virtual void split_update_score_before(state_merger*, apta_node* left, apta_node* right, tail* t);
    virtual void split_update_score_after(state_merger*, apta_node* left, apta_node* right, tail* t);
    virtual void update_score_after_recursion(state_merger*, apta_node* left, apta_node* right);

    virtual void split_update_score_before(state_merger*, apta_node* left, apta_node* right);
    virtual void split_update_score_after(state_merger*, apta_node* left, apta_node* right);

/** Called when testing a merge
* compute the score and consistency of a merge, and reset global counters/structures
*
* influence on performance, needs to be somewhat simple */
    virtual bool compute_consistency(state_merger *, apta_node* left, apta_node* right);
    virtual double compute_score(state_merger *, apta_node* left, apta_node* right);

    virtual bool split_compute_consistency(state_merger *, apta_node* left, apta_node* right);
    virtual double split_compute_score(state_merger *, apta_node* left, apta_node* right);

    virtual bool sink_convert_consistency(state_merger *, apta_node* left, apta_node* right);

    virtual void reset(state_merger *);
    virtual void reset_split(state_merger *, apta_node *);

/** Called after an update,
* when a merge has been performed successfully
* updates the structures used for computing heuristics/consistency
*
* not called when testing merges, can therefore be somewhat complex
* without a huge influence on performance*/
    virtual void update(state_merger *);

/** Called after initialization of the APTA,
* creates structures and initializes values used for computing heuristics/consistency
*
* called only once for every run, can be complex */
    virtual void initialize_after_adding_traces(state_merger *);

/** Called after reading the inputdata,
* used for creating global structures and initializes values used for computing heuristics/consistency
*
* called only once before running the algorithm, can be complex */
    virtual void initialize_before_adding_traces();
    int merge_depth_score(apta_node *left, apta_node *right);
    bool merge_same_depth(apta_node *left, apta_node *right);
    bool merge_no_root(apta_node *left, apta_node *right);
    bool merge_no_final(apta_node *left, apta_node *right);
    virtual double compute_global_score(state_merger*);
    virtual double compute_partial_score(state_merger*);

    void set_context(state_merger*);

    virtual bool pre_consistent(state_merger *merger, apta_node *left, apta_node *right);
};


#endif /* _EVALUATE_H_ */
