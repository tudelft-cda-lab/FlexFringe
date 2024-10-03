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
#include "paul_heuristic.h"

#include "base_teacher.h"
#include "common_functions.h"
#include "input_file_oracle.h"
#include "input_file_sul.h"

#include "greedy.h"
#include "inputdata.h"
#include "main_helpers.h"
#include "mem_store.h"
#include "parameters.h"
#include "state_merger.h"

#include <list>
#include <stack>
#include <queue>
#include <unordered_set>

// for the threading
#include <functional>
#include <thread>
#include "inputdatalocator.h"

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Used for multithreading.
 */
void search_instance::operator()(promise<bool>&& out, std::unique_ptr<state_merger>& merger, std::unique_ptr<apta>& the_apta, std::unique_ptr<base_teacher>& teacher, apta_node* red_node, apta_node* blue_node){
    //out.set_value(ii_handler->check_consistency(the_apta, teacher, red_node, blue_node));

    static inputdata& id = *inputdata_locator::get(); 

    auto left_access_trace = red_node->get_access_trace();
    const active_learning_namespace::pref_suf_t left_prefix = left_access_trace->get_input_sequence(true, false);

/*     static bool ii_converged = false;
    static vector< vector<int> > m_suffixes;

    if(!ii_converged && ii_handler->get_size() > 400){
        static mutex m;
        lock_guard<mutex> lock(m);
        if(m_suffixes.size()==0){
            optional< vector<int> > suffix = ds_ptr->next();
            while(suffix){
                m_suffixes.push_back(move(suffix));
                suffix = ds_ptr->next();
            }
            ii_converged = true;
        }
    }

    if() */
    
    vector< vector<int> > queries;
    optional< vector<int> > suffix = ds_ptr->next();
    vector<int> predictions;
    
    unordered_set<int> no_pred_idxs;

    while(suffix){
        vector<int> left_sequence(left_prefix.size() + suffix.value().size()); 
        left_sequence.insert(left_sequence.end(), left_prefix.begin(), left_prefix.end());
        left_sequence.insert(left_sequence.end(), suffix.value().begin(), suffix.value().end());
        
        //auto left_sequence = concat_prefsuf(left_prefix, suffix.value());
        if(left_sequence.size() > MAX_LEN){
            no_pred_idxs.insert(queries.size()+no_pred_idxs.size());
            suffix = ds_ptr->next();
            continue;
        }
        
        queries.push_back(move(left_sequence));
        if(queries.size() >= MIN_BATCH_SIZE){
            m_mutex.lock();
            vector< pair<int, float> > answers = teacher->ask_type_confidence_batch(queries, *(inputdata_locator::get()));
            m_mutex.unlock();
            int answers_idx = 0;

            for(int i=0; i<answers.size()+no_pred_idxs.size(); ++i){
                if(no_pred_idxs.contains(i)){
                    predictions.push_back(-1);
                    continue;
                }
                auto& pred = answers[answers_idx];
                predictions.push_back(pred.first);
                ++answers_idx;
            }

            queries.clear();
            no_pred_idxs.clear();
        }

        suffix = ds_ptr->next();
    }

    if(queries.size() > 0){
        m_mutex.lock();
        vector< pair<int, float> > answers = teacher->ask_type_confidence_batch(queries, *(inputdata_locator::get()));
        m_mutex.unlock();
        int answers_idx = 0;

        for(int i=0; i<answers.size()+no_pred_idxs.size(); ++i){
        if(no_pred_idxs.contains(i)){
            predictions.push_back(-1);
            continue;
        }
        auto& pred = answers[answers_idx];
        predictions.push_back(pred.first);
        ++answers_idx;
        }
    }

    const auto& memoized_predictions = static_cast<distinguishing_sequence_fill*>(ii_handler.get())->get_memoized_predictions();
    if(memoized_predictions.size() != predictions.size()){
        cerr << "Something weird happened." << endl;
    }

    int agreed = 0;
    int disagreed = 0;
    for(int i=0; i<predictions.size(); ++i){
        if(memoized_predictions[i]==-1 || predictions[i]==-1)
            continue;
        else if(memoized_predictions[i] == predictions[i])
            ++agreed;
        else
            ++disagreed;
    }

    float ratio = static_cast<float>(disagreed) / (static_cast<float>(disagreed) + static_cast<float>(agreed));
    if(ratio > 0.01){
        //cout << "Disagreed: " << disagreed << " | agreed: " << agreed << " | ratio: " << ratio << endl;
        return out.set_value(false);
    }
    return out.set_value(true);
}

/**
 * Relevant for parallelization.
 */
bool paul_algorithm::merge_check(unique_ptr<ii_base>& ii_handler, unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta, unique_ptr<base_teacher>& teacher, apta_node* red_node, apta_node* blue_node){
    return ii_handler->check_consistency(the_apta, teacher, red_node, blue_node);
}

/**
 * @brief The strategy to find the best operation as explained in the paper.
 * @return refinement* The best currently possible operation according to the heuristic.
 */
refinement* paul_algorithm::get_best_refinement(unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta, unique_ptr<base_teacher>& teacher){
    const static bool ADD_TRACES = false; // if true we add sequences to the tree, else we only test, see down the code
    const static bool MERGE_WITH_LARGEST = true;
    const static int N_THREADS = 10;
    
    unordered_set<apta_node*> blue_its; // = state_set();
    state_set red_its = state_set();
    
    for (blue_state_iterator it = blue_state_iterator(the_apta->get_root()); *it != nullptr; ++it){
        auto blue_node = *it;
        if(blue_node->get_size() != 0) blue_its.insert(blue_node);
    }

    for (red_state_iterator it = red_state_iterator(the_apta->get_root()); *it != nullptr; ++it){
        auto red_node = *it;
        if(red_node->get_size() != 0) red_its.insert(red_node);
    }

    refinement_set rs;
    for (auto blue_node: blue_its) {
        
        // pre-compute on all pairs to make pre_compute of blue node consistent with all the others
        //if(ii_handler->size() > 400){
            for(apta_node* red_node: red_its){
                ii_handler->pre_compute(the_apta, teacher, red_node, blue_node);
            }
        //}


        bool mergeable = false;
        ii_handler->pre_compute(the_apta, teacher, blue_node);

        if(/* N_THREADS == 1 */false){
            for(apta_node* red_node: red_its){
                refinement* ref = merger->test_merge(red_node, blue_node);
                if(ref == nullptr) continue;

                // we only want to add data if they appear consistent so far
                if(ADD_TRACES){
                    ii_handler->complement_nodes(the_apta, teacher, red_node, blue_node);
                    ref = merger->test_merge(red_node, blue_node);
                }
                else if(!ii_handler->check_consistency(the_apta, teacher, red_node, blue_node)){
                    continue;
                }
                
                if (ref != nullptr && ref->score > 0){
                    rs.insert(ref); // TODO: should we erase the refs somewhere as well?
                    mergeable = true;
                }
                if(MERGE_WITH_LARGEST && mergeable)
                    break;
            }
        }
        else{
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
                threads.push_back(thread(std::ref(search_instances[current_refs.size()-1]), move(p), std::ref(merger), std::ref(the_apta), std::ref(teacher), red_node, blue_node));
                    
                //t_res.push_back(async(launch::async, search_instance(), std::ref(merger), std::ref(the_apta), std::ref(teacher), red_node, blue_node));
                //t_res.push_back(async(launch::async, paul_algorithm::merge_check, std::ref(ii_handler), std::ref(merger), std::ref(the_apta), std::ref(teacher), red_node, blue_node));
                
                if(t_res.size()==N_THREADS){
                    // sync threads and collect results
                    for(int i=0; i<t_res.size(); ++i){
                        threads[i].join();
                        bool merge_consistent = t_res[i].get();
                        if(merge_consistent){
                            rs.insert(current_refs[i]);
                            mergeable = true;
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
            }
        }
        
        if(!mergeable)
            rs.insert(mem_store::create_extend_refinement(merger.get(), blue_node));
    }

    refinement *r = nullptr;
    if (!rs.empty()) {
        r = *(rs.begin());
        for(auto it = rs.begin(); it != rs.end(); ++it){
            auto rf = *it;
            if(r != rf) rf->erase();
        }
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
list<refinement*> paul_algorithm::find_hypothesis(list<refinement*>& previous_refs, unique_ptr<state_merger>& merger, unique_ptr<apta>& the_apta, unique_ptr<base_teacher>& teacher) {
    list<refinement*> performed_refs;
    if(previous_refs.size() > 0)
        performed_refs = retry_merges(previous_refs, merger, the_apta);

    refinement* best_ref = paul_algorithm::get_best_refinement(merger, the_apta, teacher);
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
        best_ref = paul_algorithm::get_best_refinement(merger, the_apta, teacher);

        num++;
    }

    return performed_refs;
}

/**
 * @brief Add the counterexample to the tree. Newly creatd states will be queried.
 */
void paul_algorithm::proc_counterex(const unique_ptr<base_teacher>& teacher, inputdata& id,
                                      unique_ptr<apta>& the_apta, const vector<int>& counterex,
                                      unique_ptr<state_merger>& merger, const refinement_list refs) const {
    
    active_learning_namespace::reset_apta(merger.get(), refs);
    vector<int> substring;
    apta_node* n = the_apta->get_root();
    for (auto s : counterex) {
        substring.push_back(s);
        trace* parse_trace = vector_to_trace(substring, id, 0); // TODO: inefficient like this, 0 is a dummy type that does not matter
        tail* t = substring.size() == 0 ? parse_trace->get_end() : parse_trace->get_end()->past_tail;
        apta_node* n_child = active_learning_namespace::get_child_node(n, t);

        if (n_child == nullptr) {
            auto access_trace = n->get_access_trace();
            pref_suf_t seq;
            seq = access_trace->get_input_sequence(true, true);

            vector< vector<int> > query;
            query.push_back(move(seq));
            vector< pair<int, float> > res = teacher->ask_type_confidence_batch(query, id);

            int reverse_type  = res[0].first;
            float confidence = res[0].second;

            trace* new_trace = active_learning_namespace::vector_to_trace(seq, id, reverse_type);
            id.add_trace_to_apta(new_trace, the_apta.get(), false);

            paul_data* data;
            data = dynamic_cast<paul_data*>(n->get_data());
            data->set_confidence(confidence);

            n_child = active_learning_namespace::get_child_node(n, t);
        }

        n = n_child;
        mem_store::delete_trace(parse_trace);
    }
}

void paul_algorithm::run(inputdata& id) {
    int n_runs = 0;

#ifndef NDEBUG
    LOG_S(INFO) << "Creating debug directory " << DEBUG_DIR;
    std::filesystem::create_directories(DEBUG_DIR);
#endif
    LOG_S(INFO) << "Running PAUL algorithm.";

    auto eval = unique_ptr<evaluation_function>(get_evaluation());
    eval->initialize_before_adding_traces();

    auto the_apta = unique_ptr<apta>(new apta());
    auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));
    this->oracle->initialize(merger.get());
    the_apta->set_context(merger.get());
    eval->set_context(merger.get());
    
    id.add_traces_to_apta(the_apta.get());
    eval->initialize_after_adding_traces(merger.get());

    const vector<int> alphabet = id.get_alphabet();
    cout << "Alphabet: ";
    active_learning_namespace::print_sequence<vector<int>::const_iterator>(alphabet.cbegin(), alphabet.cend());

    list<refinement*> performed_refs;
    while(ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS){
        performed_refs = find_hypothesis(performed_refs, merger, the_apta, teacher);

        optional<pair<vector<int>, int>> query_result = oracle->equivalence_query(merger.get(), teacher);
        if (!query_result) {
            cout << "Found consistent automaton => Print." << endl;
            print_current_automaton(merger.get(), OUTPUT_FILE, ".final"); // printing the final model each time

            for(auto ref: performed_refs)
                mem_store::delete_refinement(ref);

            return;
        }

        const int type = query_result.value().second;
        const vector<int>& cex = query_result.value().first;

        cout << "Counterexample of length " << cex.size() << " found: ";
        for(auto s: cex)
            cout << id.get_symbol(s) << " ";
        cout << endl;        
        proc_counterex(teacher, id, the_apta, cex, merger, performed_refs);

        ++n_runs;
    }

    for(auto ref: performed_refs)
        mem_store::delete_refinement(ref);

    cout << "Reached maximum number of runs. Printing current hypothesis and terminating." << endl;
    print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
}
