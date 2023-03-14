// TODO: those are a lot of includes. Do we need them all?
#include "stream.h"
#include <cstdlib>

#include "greedy.h"
#include "state_merger.h"
#include "evaluate.h"
#include "dfasat.h"
#include "evaluation_factory.h"
#include "searcher.h"
#include "parameters.h"

#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stack>

#include <chrono> // for measuring performance
#include <fstream> // for measuring performance

const bool RETRY_MERGES = true;
const bool PRINT_ALL_MODELS = false; // for debugging purposes
const bool EXPERIMENTAL_RUN = true;

int P_PERCENT = 0; // to get each i-th percentage. We print the model then

void stream_object::greedyrun_no_undo(state_merger* merger, const int seq_nr, const bool last_sequence, const int n_runs){
    static int count_printouts = 1;

/*     if(PRINT_ALL_MODELS){
      merger->todot();
      std::ostringstream oss2;
      oss2 << "model_before_" << this->batch_number << ".dot";
      ofstream output1(oss2.str().c_str());
      output1 << merger->dot_output;
      output1.close();
    } */

    int c2 = 0;

    refinement* top_ref;
    top_ref = merger->get_best_refinement();
    while(top_ref != 0){
      top_ref->doref(merger);
      top_ref = merger->get_best_refinement();

      if(PRINT_ALL_MODELS){
        merger->todot();
        std::ostringstream oss2;
        oss2 << "batch_" << this->batch_number << "_" << c2 << ".dot";
        ofstream output1(oss2.str().c_str());
        output1 << merger->dot_output;
        output1.close();

        ++c2;
      }
    }

/*     if(PRINT_ALL_MODELS){
      merger->todot();
      std::ostringstream oss2;
      oss2 << "model_after_" << this->batch_number << ".dot";
      ofstream output1(oss2.str().c_str());
      output1 << merger->dot_output;
      output1.close();
    } */

    if(seq_nr > count_printouts * P_PERCENT || last_sequence){
      cout << "Processed " << count_printouts * 10 << " percent" << endl;
      ++count_printouts;
    }
}

void stream_object::greedyrun_retry_merges(state_merger* merger, const int seq_nr, const bool last_sequence, const int n_runs){
    static int count_printouts = 1;

    static int c = 0;
    int c2 = 0;

    stack< refinement* > refinement_stack;
    queue<refinement*> failed_refs;

    static ofstream tree_size_doc("tree_size_per_batch.txt");
    refinement* top_ref;

    this->states_to_append_to.clear();

    if(PRINT_ALL_MODELS){
      merger->todot();
      std::ostringstream oss2;
      oss2 << "model_before_" << this->batch_number << ".dot";
      ofstream output1(oss2.str().c_str());
      output1 << merger->dot_output;
      output1.close();
    }

    // for tracking the refinements
    static ofstream outf;
    string recomputed_merge = "nr"; // nr for no_recompute, r for recompute
    bool track_refinements = (n_runs == 47654 || n_runs == 47655 || n_runs == 46066 || n_runs == 46067 || n_runs == 51696 || n_runs == 51695);
    if(track_refinements){
      std::ostringstream oss; // for tracking refinements (debugging)
      oss << "refinements_at_" << n_runs << "_full.txt";
      outf.open(oss.str().c_str());
    }

    if(currentrun->empty()){
      top_ref = merger->get_best_refinement();
    }
    else{
        top_ref = currentrun->front();
        currentrun->pop_front();
        //if(top_ref->testref(merger) == false){
          // nothing
        //}
    }

    // TODO: delete unused refinements
    while(top_ref != 0){
        if(top_ref->test_ref_structural(merger)){
          if(top_ref->test_ref_consistency(merger)){
            nextrun->push_back(top_ref);
            refinement_stack.push(top_ref);
            top_ref->doref(merger);

            if(PRINT_ALL_MODELS){
              merger->todot();
              std::ostringstream oss2;
              oss2 << "batch_" << this->batch_number << "_" << c2 << ".dot";
              ofstream output1(oss2.str().c_str());
              output1 << merger->dot_output;
              output1.close();

              ++c2;
            }

            //if(track_refinements){
            //  outf << top_ref->get_string(merger) << ", " << "or" << endl; // or for old-refinement
            //}
          }
        }
        else{
          failed_refs.push(top_ref);
        }

        if(currentrun->empty()){
            top_ref = 0;
        }
        else{
            top_ref = currentrun->front();
            currentrun->pop_front();
        }
    }

    top_ref = merger->get_best_refinement();
    while(top_ref != 0){
      nextrun->push_back(top_ref);
      refinement_stack.push(top_ref);
      top_ref->doref(merger);

      if(PRINT_ALL_MODELS){
        merger->todot();
        std::ostringstream oss2;
        oss2 << "batch_" << this->batch_number << "_" << c2 << ".dot";
        ofstream output1(oss2.str().c_str());
        output1 << merger->dot_output;
        output1.close();

        ++c2;
      }

      queue<refinement*> tmp;
      while(!failed_refs.empty()){
        top_ref = failed_refs.front();
        failed_refs.pop();
        if(top_ref->test_ref_structural(merger)){
          if(top_ref->test_ref_consistency(merger)){
            nextrun->push_back(top_ref);
            refinement_stack.push(top_ref);
            top_ref->doref(merger);

            if(PRINT_ALL_MODELS){
              merger->todot();
              std::ostringstream oss2;
              oss2 << "batch_" << this->batch_number << "_" << c2 << ".dot";
              ofstream output1(oss2.str().c_str());
              output1 << merger->dot_output;
              output1.close();

              ++c2;
            }

            //if(track_refinements){
            //  outf << top_ref->get_string(merger) << ", " << "fr" << endl;
            //}
          }
        }
        else{
          tmp.push(top_ref);
        }
      }
      failed_refs = tmp;

      top_ref = merger->get_best_refinement();
      //if(track_refinements && top_ref != 0){
      //  outf << top_ref->get_string(merger) << ", " << "gbr" << endl;
      //}
    }

    if(PRINT_ALL_MODELS){
      merger->todot();
      std::ostringstream oss2;
      oss2 << "model_after_" << this->batch_number << ".dot";
      ofstream output1(oss2.str().c_str());
      output1 << merger->dot_output;
      output1.close();
    }

    if(seq_nr > count_printouts * P_PERCENT || last_sequence){
      cout << "Processed " << count_printouts * 10 << " percent" << endl;
      ++count_printouts;
    }

    // undo the merges
    while(!refinement_stack.empty()){
      top_ref = refinement_stack.top();
      top_ref->undo(merger);
      refinement_stack.pop();

      states_to_append_to.insert(top_ref->red->get_number());
      if(top_ref->type() == refinement_type::merge_rf_type){
        states_to_append_to.insert(dynamic_cast<merge_refinement*>(top_ref)->blue->get_number());
      }
    }

    if(track_refinements){
      outf.close();
    }

    delete currentrun;
    currentrun = nextrun;
    nextrun = new refinement_list();
}

template<typename T>
void print_time(const T start_time, const T batch_start_time, const unsigned int n_runs, ofstream& time_doc){
  stringstream time_string;
  auto end_time = std::chrono::system_clock::now();

  auto total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  auto total_duration_s = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
  auto total_duration_min = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time).count();
  auto total_duration_h = std::chrono::duration_cast<std::chrono::hours>(end_time - start_time).count();

  auto batch_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - batch_start_time).count();
  auto batch_duration_s = std::chrono::duration_cast<std::chrono::seconds>(end_time - batch_start_time).count();
  auto batch_duration_min = std::chrono::duration_cast<std::chrono::minutes>(end_time - batch_start_time).count();
  auto batch_duration_h = std::chrono::duration_cast<std::chrono::hours>(end_time - batch_start_time).count();

  time_string << "Runs:" << n_runs << " Total(ms):" << total_duration_ms << " Total(s):" << total_duration_s << " Total(min):" << total_duration_min << " Total(h):" << total_duration_h << " Batch(ms):" << batch_duration_ms << " Batch(s):" << batch_duration_s << " Batch(min):" << batch_duration_min << " Batch(h):" << batch_duration_h << endl;
  time_doc << time_string.str();
}

/**
 * @brief Runs the stream mode.
 * 
 * This is the original version of the streaming function, where we can toggle the old and the new version.
 * 
 * @param merger The selected state merger instance.
 * @param param The global parameters.
 * @param input_stream The input data file.
 * @param id Input-data wrapper object. 
 * @return int 
 */
int stream_object::stream_mode(state_merger* merger, ifstream& input_stream, inputdata* id, parser* input_parser) {
    const int BATCH_SIZE = 500; // it was 500 in the original experiments
    unsigned int seq_nr = 0;
    bool last_sequence = false;

    P_PERCENT = static_cast<int>(id->get_max_sequences() / 10); // to track the percent
    cout << "P_PERCENT: " << P_PERCENT << endl;

//    if(!CONVERT_RAW_DATA) id->read_abbadingo_header(input_stream);

    ofstream time_doc("times_per_batch.txt");

    // for performance measurement
    unsigned int n_runs = 0;
    auto start_time = std::chrono::system_clock::now();

    while(true) {
      //merger->reset(); // TODO: should we use this?
      int read_lines = 0;
      auto batch_start_time = std::chrono::system_clock::now();
      while (read_lines < BATCH_SIZE){
        if(input_stream.eof()){
          last_sequence = true;
          break; // TODO: for experiments, delete afterwards
        }

        ++read_lines;
        ++seq_nr;
        
        std::optional<trace*> trace_opt = id->read_trace(*input_parser, *parser_strategy);

        if(EXPERIMENTAL_RUN && !trace_opt){
          last_sequence = true;
          break;
        }
        else{
          while(!trace_opt) trace_opt = id->read_trace(*input_parser, *parser_strategy); // making sure we got a value; for real streaming
        }
        trace* new_trace = trace_opt.value();
        new_trace->sequence = seq_nr;
        
        id->add_trace_to_apta(new_trace, merger->get_aut(), this->states_to_append_to);
        if(!ADD_TAILS) new_trace->erase();
      }

      if(RETRY_MERGES) greedyrun_retry_merges(merger, seq_nr, last_sequence, n_runs);
      else greedyrun_no_undo(merger, seq_nr, last_sequence, n_runs);

      print_time(start_time, batch_start_time, n_runs, time_doc);
      time_doc.flush();
      ++(this->batch_number);
      ++n_runs;

      if(input_stream.eof()){
        if(RETRY_MERGES) {
          greedyrun_retry_merges(merger, seq_nr, last_sequence, n_runs);

          if(currentrun->size() == 0){
            cerr << "Error: Currentrun object empty after running the whole input. No output to be generated." << endl;
            throw new exception;
          }

          for(auto top_ref: *currentrun){
              top_ref->doref(merger);
          }
        }

        cout << "Finished parsing file. End of program." << endl;
        time_doc.close();
        return 0;
      }
    }
    return 0;
}
