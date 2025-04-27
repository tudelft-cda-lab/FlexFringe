/**
 * @file paul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-08-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "paul.h"
#include "common_functions.h"

#include "inputdatalocator.h"
#include "abbadingoparser.h"
#include "csvparser.h"
#include "output_manager.h"

#include "inputdata.h"
#include "common.h"
#include "mem_store.h"
#include "parameters.h"
#include "state_merger.h"

#include <list>

// for the threading
#include <functional>
#include <thread>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Checks if node's data has has need for update, i.e. if the set of distinguishing sequences has increased, and if yes then
 * update the predictions of the node.
 */
void paul_algorithm::update_node_data(apta_node* n, std::unique_ptr<apta>& aut) const {
    auto* n_data = get_node_data(n);

    if(!n_data->has_type()){
        ii_handler->complete_node(n, aut);
    }

    if(n_data->get_predictions().size() != ii_handler->size()){
        auto y_pred = ii_handler->predict_node_with_sul(*aut, n);
        n_data->set_predictions(std::move(y_pred));
    }
}

/**
 * @brief For convenience. Gets data from node and converts to PAUL data. This works because PAUL algorithm is deeply intertwined with 
 * its own heuristic.
 */
paul_data* paul_algorithm::get_node_data(apta_node* n) const {
    return dynamic_cast<paul_data*>(n->get_data());
}

/**
 * @brief Checks if blue node has any merge partner among the red nodes, given by red_its. 
 * If blue node does not find a merge partner, then return a nullptr.
 */
refinement* paul_algorithm::check_blue_node_for_merge_partner(apta_node* const blue_node, unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta,
                                                              const state_set& red_its){
    constexpr static int N_THREADS = 1;
    const static bool MEMOIZE_PREDICTIONS = AL_SAVE_RUNTIME_FOR_SPACE;

    if(MEMOIZE_PREDICTIONS){
        update_node_data(blue_node, the_apta);
    }
    else{
        for(apta_node* red_node: red_its){
            ii_handler->pre_compute(the_apta, red_node, blue_node);
        }
        ii_handler->pre_compute(the_apta, blue_node);
    }

    refinement_set rs;
    bool mergeable = false;
    if(N_THREADS == 1){
        for(apta_node* red_node: red_its){
            refinement* ref = merger->test_merge(red_node, blue_node);
            if(ref == nullptr){
                continue;
            }

            // compare the nodes based on the SUL's predictions
            if(MEMOIZE_PREDICTIONS){
                update_node_data(red_node, the_apta);
                if(!ii_handler->distributions_consistent(get_node_data(blue_node)->get_predictions(), get_node_data(red_node)->get_predictions())){
                    continue;
                }
            }
            else{
                // we only want to add data if they appear consistent so far
                if(!ii_handler->check_consistency(the_apta, red_node, blue_node)){
                    continue;
                }
            }
                
            ref->score = ii_handler->get_score(); // score computed in check_consistency() or distributions_consistent()
            if(ref->score > 0){
                rs.insert(ref);
                mergeable = true;
            }
            else{
                mem_store::delete_refinement(ref);
            }

            if(MERGE_WITH_LARGEST && mergeable){
                //return *(rs.begin());
            }
        }
    }
    else{/*
            static vector<search_instance> search_instances(N_THREADS); // avoid redundant reconstruction of objects

            vector<thread> threads;
            vector<future<bool>> t_res;
            vector<refinement*> current_refs;

            for(apta_node* red_node: red_its){
                refinement* ref = merger->test_merge(red_node, blue_node);
                if(ref == nullptr) continue;
                current_refs.push_back(ref);

                promise<bool> p;
                t_res.push_back(p.get_future());
                threads.push_back(thread(std::ref(search_instances[current_refs.size()-1]), move(p), std::ref(merger), std::ref(the_apta), std::ref(oracle), red_node, blue_node));
                    
                //t_res.push_back(async(launch::async, search_instance(), std::ref(merger), std::ref(the_apta), std::ref(oracle), red_node, blue_node));
                //t_res.push_back(async(launch::async, paul_algorithm::merge_check, std::ref(ii_handler), std::ref(merger), std::ref(oracle), std::ref(the_apta), red_node, blue_node));
                
                if(t_res.size()==N_THREADS){
                    // sync threads and collect results
                    for(int i=0; i<t_res.size(); ++i){
                        threads[i].join();
                        bool merge_consistent = t_res[i].get();
                        if(merge_consistent){
                            rs.insert(current_refs[i]);
                            mergeable = true;
                        }
                        else{
                            mem_store::delete_refinement(current_refs[i]);
                        }
                    }
                    current_refs.clear();
                    t_res.clear();
                    threads.clear();

                    if(MERGE_WITH_LARGEST && mergeable)
                        break;
                }
            }

            // we need to join the remaining threads to not cause an exception (or alternatively roll out a try block, which is probably more expensive?)
            for(int i=0; i<threads.size(); ++i){
                threads[i].join();
                bool merge_consistent = t_res[i].get();
                if(merge_consistent){
                    rs.insert(current_refs[i]);
                    mergeable = true;
                }
            }*/
        }
    
    refinement* r = nullptr;
    if (!rs.empty()){
        r = *(rs.begin());
        for(auto it = rs.begin(); it != rs.end(); ++it){
            auto rf = *it;
            if(r != rf) rf->erase();
        }
    }

    return r;
}

/**
 * @brief The strategy to find the best operation as explained in the paper.
 * @return refinement* The best currently possible operation according to the heuristic.
 */
refinement* paul_algorithm::get_best_refinement(unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta){

    state_set red_its = state_set();
    unordered_set<apta_node*> blue_its;
    static unordered_set<apta_node*> non_mergeable_blue_its;

    for (blue_state_iterator it = blue_state_iterator(the_apta->get_root()); *it != nullptr; ++it){
        auto blue_node = *it;
        if(blue_node->get_size() != 0 && !non_mergeable_blue_its.contains(blue_node)) blue_its.insert(blue_node);
    }

    for (red_state_iterator it = red_state_iterator(the_apta->get_root()); *it != nullptr; ++it){
        auto red_node = *it;
        if(red_node->get_size() != 0) red_its.insert(red_node);
    }

/*     {
        cout << "\nSize blue nodes: " << blue_its.size() << "\n";
        cout << "Size red nodes: " << red_its.size() << "\n" << endl;
    } */

    refinement_set rs;
    for (auto blue_node: blue_its) {
        static const bool MERGE_WITH_LARGEST_BLUE = MERGE_MOST_VISITED;
        refinement* ref = check_blue_node_for_merge_partner(blue_node, merger, the_apta, red_its);
        
        if(ref == nullptr){
            rs.insert(mem_store::create_extend_refinement(merger.get(), blue_node));
            non_mergeable_blue_its.insert(blue_node);
        }
        else if(MERGE_WITH_LARGEST_BLUE)
            return ref;
        else
            rs.insert(ref);
    }

    unordered_set<apta_node*> mergeable_nodes;
    for (auto blue_node: non_mergeable_blue_its) {
        refinement* ref = check_blue_node_for_merge_partner(blue_node, merger, the_apta, red_its);
        
        if(ref == nullptr){
            rs.insert(mem_store::create_extend_refinement(merger.get(), blue_node));
            non_mergeable_blue_its.insert(blue_node);
        }
        else if(MERGE_WITH_LARGEST){
            non_mergeable_blue_its.erase(blue_node);
            return ref;
        }
        else{
            rs.insert(ref);
            mergeable_nodes.insert(blue_node);
        }
    }

    for(auto blue_node: mergeable_nodes){
        non_mergeable_blue_its.erase(blue_node);
    }

    refinement *r = nullptr;
    if (!rs.empty()) {
        r = *(rs.begin());
        for(auto it = rs.begin(); it != rs.end(); ++it){
            auto rf = *it;
            if(r != rf) rf->erase();
        }
    }

    if(r!=nullptr && non_mergeable_blue_its.contains(r->red)){
        non_mergeable_blue_its.erase(r->red);
    }

    return r;
}

/**
 * @brief Retries the merges that we did in the last run. This method leans on the streaming approach that we 
 * have already.
 * 
 * Side effect: Refinements that cannot be done anymore will get deleted (avoiding memory leaks).
 * 
 * @return list<refinement*> 
 */
list<refinement*> paul_algorithm::retry_merges(list<refinement*>& previous_refs, unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta) {
    list<refinement*> performed_refs;

    for(auto& ref: previous_refs){
        if(ref->test_ref_structural(merger.get()) && ref->test_ref_consistency(merger.get())){
            ref->doref(merger.get());
            performed_refs.push_back(move(ref));
        }
        else{
            mem_store::delete_refinement(ref);
        }
    }

    return performed_refs;
}

/**
 * @brief Does one minimization step and outputs a hypothesis.
 * 
 * @param previous_refs Refinements that have already been done.
 * @return list<refinement*> A list of performed refinements.
 */
list<refinement*> paul_algorithm::find_hypothesis(list<refinement*>& previous_refs, unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta) {
    list<refinement*> performed_refs;
    if(previous_refs.size() > 0)
        performed_refs = retry_merges(previous_refs, merger, the_apta);

    refinement* best_ref = paul_algorithm::get_best_refinement(merger, the_apta);
    int num = 0;
    while(best_ref != nullptr){
        cout << " ";
        best_ref->print_short();
        cout << " ";
        std::cout.flush();

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << num;
        std::string s = ss.str();

/*         {
            static int c = 0;
            merger->print_dot("before_" + to_string(c++) + ".dot");
        } */

        best_ref->doref(merger.get());
        performed_refs.push_back(best_ref);
/*         if(dynamic_cast<merge_refinement*>(best_ref) != nullptr){
            auto c_ref = dynamic_cast<merge_refinement*>(best_ref);
            cout << "Merge of " << c_ref->red->get_number() << " and " << c_ref->blue->get_number() << endl;
        } */

//#ifndef NDEBUG
/*         {
            static int c = 0;
            merger->print_dot("after_" + to_string(c++) + ".dot");
        } */
//#endif

        //delete best_ref;
        best_ref = paul_algorithm::get_best_refinement(merger, the_apta);
        num++;
    }

    return performed_refs;
}

/**
 * @brief Add the counterexample to the tree. Newly creatd states will be queried.
 */
void paul_algorithm::proc_counterex(inputdata& id, unique_ptr<apta>& the_apta, const vector<int>& counterex,
                                      unique_ptr<state_merger>& merger, const refinement_list refs) const {
    
    active_learning_namespace::reset_apta(merger.get(), refs);
    vector<int> substring;
    apta_node* n = the_apta->get_root();

    for (auto s : counterex) {
        substring.push_back(s);
        trace* parse_trace = vector_to_trace(substring, id, 0); // TODO: inefficient like this, since we redo traces from scratch again. Sidenote: 0 is a dummy type that does not matter
        tail* t = parse_trace->get_end()->past_tail;
        apta_node* n_child = active_learning_namespace::get_child_node(n, t);

        if (n_child == nullptr) {

            vector< vector<int> > query(1);
            query[0] = substring;
            const sul_response res = oracle->ask_sul(query, id);

            int reverse_type = res.GET_INT_VEC()[0];
            double confidence = res.GET_DOUBLE_VEC()[0];
            assert(res.GET_INT_VEC().size() == 1);

            trace* new_trace = active_learning_namespace::vector_to_trace(substring, id, reverse_type);
            id.add_trace_to_apta(new_trace, the_apta.get(), false);
            n_child = active_learning_namespace::get_child_node(n, t);
            assert(n_child != nullptr);

            auto* data = get_node_data(n_child);
            if(!data->has_type()) [[likely]] { // should always be true
                data->set_confidence(confidence);
                data->add_inferred_type(reverse_type);
            }
        }

        n = n_child;
        mem_store::delete_trace(parse_trace);
    }
}

void paul_algorithm::run(inputdata& id) {
    int n_runs = 0;
    cout << "Running PAUL algorithm." << endl;

    auto eval = unique_ptr<evaluation_function>(get_evaluation());
    eval->initialize_before_adding_traces();

    auto the_apta = unique_ptr<apta>(new apta());
    auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));
    this->oracle->initialize(merger.get());
    the_apta->set_context(merger.get());
    eval->set_context(merger.get());
    
    id.add_traces_to_apta(the_apta.get());
    eval->initialize_after_adding_traces(merger.get());

    const auto& rtypes = id.get_r_types();
    cout << "\nType prediction mapping is as follows:\n";
    for(auto& [k, v]: rtypes){
        cout << "(k,v) - " << k << " : " << v << "\n";
    }
    cout << endl;

    {    
        int n_states = 0;
        for (APTA_iterator it = APTA_iterator(the_apta->get_root()); *it != nullptr; ++it){
            ++n_states;
        }
        cout << "Size of raw APTA: " << n_states << endl;
    }

    cout << "Initializing ii_handler" << endl;
    ii_handler->initialize(the_apta); // must happen after traces have been added to apta!
    try{
        dynamic_cast<paul_heuristic*>(eval.get())->provide_ii_handler(ii_handler);    
    }
    catch(...){
        throw invalid_argument("Cannot provide heuristic with ii_handler. Using paul heuristic?");
    }
    cout << "ii_handler is initialized" << endl;

    const vector<int> alphabet = id.get_alphabet();
    cout << "Alphabet: ";
    active_learning_namespace::print_sequence<vector<int>::const_iterator>(alphabet.cbegin(), alphabet.cend());

    list<refinement*> performed_refs;
    while(ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS){
        performed_refs = find_hypothesis(performed_refs, merger, the_apta);
        cout << "Found hypothesis. Now testing" << endl;

        {
            static int model_nr = 0;
            cout << "printing model " << model_nr  << endl;

            output_manager::print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_refs");
            output_manager::print_final_automaton(merger.get(), "." + to_string(model_nr) + ".pre_final");
        }

        optional<pair<vector<int>, sul_response>> query_result = oracle->equivalence_query(merger.get());
        if (!query_result) {
            cout << "Found consistent automaton => Print." << endl;
            output_manager::print_final_automaton(merger.get(), ".final");

            for(auto ref: performed_refs)
                mem_store::delete_refinement(ref);

            return;
        }

        const int type = query_result.value().second.get_int();
        const vector<int>& cex = query_result.value().first;

        cout << "Counterexample of length " << cex.size() << " found: ";
        for(auto s: cex)
            cout << id.get_symbol(s) << " ";
        cout << endl;        
        proc_counterex(id, the_apta, cex, merger, performed_refs);

/*         {
            static int model_nr = 0;
            cout << "printing model " << model_nr  << endl;
            output_manager::print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_cex");
        } */

        ++n_runs;
    }

    for(auto ref: performed_refs)
        mem_store::delete_refinement(ref);

    cout << "Reached maximum number of runs. Printing current hypothesis and terminating." << endl;
    output_manager::print_final_automaton(merger.get(), ".final");
}

void paul_algorithm::load_inputdata(){
    ifstream input_stream(INPUT_FILE);
    cout << "Input file: " << INPUT_FILE << endl;
    if (!input_stream) {
        cerr << "Input file not found, aborting" << endl;
        exit(-1);
    } else {
        cout << "Using input file: " << INPUT_FILE << endl;
    }

    unique_ptr<parser> input_parser; 
    if (INPUT_FILE.ends_with(".csv")) {
        input_parser = make_unique<csv_parser>(input_stream, csv::CSVFormat().trim({' '}));
    } else {
        input_parser = make_unique<abbadingoparser>(input_stream);
    }

    cout << "Loading input data into apta" << endl;
    inputdata* id = inputdata_locator::get();
    id->read(input_parser.get());
    input_stream.close();
    cout << "Loaded input data into apta" << endl;
}