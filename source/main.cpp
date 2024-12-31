/**
 * @file main.cpp
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "running_mode_factory.h"
#include "utility/loguru.hpp"
#include "CLI11.hpp"

#include "parameters.h"
#include "output_manager.h"

#include "misc/sqldb.h"
#include "misc/utils.h"

#include <cstdlib>
#include <ranges>
#include <iostream>
#include <string>
#include <memory>

std::string COMMAND; // TODO: Is this necessary? Put it somewhere else, e.g. common.h or output_manager.h

/**
 * @brief Main run method. Branches out based on the type of session to run.
 */
void run() {
    output_manager::init_outfile_path();
    std::unique_ptr<running_mode_base> mode = running_mode_factory::get_mode();
    mode->run();

    // different modes can have different types of output (or possibly none in case of interactive mode)
    mode->generate_output();
    std::cout << "Finished running successfully. Ending program." << std::endl;
}

/**
 * @brief Main method. Reads in arguments and starts application 
 * by running "run()" function with the set of parsed parameters.
 * 
 */
#ifndef UNIT_TESTING
int main(int argc, char *argv[]){

    for(int i = 0; i < argc; i++) {
      COMMAND_LINE += std::string(argv[i]) + std::string(" ");
    }

    //cout << "welcome, running git commit " << gitversion <<  " with: "<< param->command << endl;

    // CLI11-based parameter parsing to replace libpopt
    // https://cliutils.gitlab.io/CLI11Tutorial/
    CLI::App app{"flexFringe"
                 "Copyright 2022 Sicco Verwer, Delft University of Technology"
                 "with contributions from Christian Hammerschmidt, Delft University of Technology,\
                  University of Luxembourg"
                 "with contributions from APTA Technologies B.V."
                 "based on"
                 "DFASAT with random greedy preprocessing"
                 "Copyright 2015 Sicco Verwer and Marijn Heule, Delft University of Technology."
    };

    // remove -h short-form from help because for backward-compatibility --heuristic needs nit
    app.set_help_flag("--help", "Print this help message and exit");

    // read parameters from ini file if desired
    std::string default_file_name = "flexfringe.ini";
    app.add_option("tracefile", INPUT_FILE, "Name of the input file containing the traces, either in Abbadingo or JSON format.")->required();
    app.add_option("--outputfile", OUTPUT_FILE, "The prefix of the output file name. Default is same as input.");
    app.add_option("--output", OUTPUT_TYPE, "Switch between output in dot, json, or both (default) formats.");
    app.add_option("--logpath", LOG_PATH, "The path to write the flexfringe log file to. Defaults to \"flexfringe.log\"");
    app.add_option("--debugdir", DEBUG_DIR, "The dir to write any debug data.");
    app.set_config("--ini", default_file_name, "Read an ini file", false);
    app.add_option("--mode", OPERATION_MODE, "batch (default), interactive, predict, active_learning, or stream, depending on the mode of operation.");
    app.add_option("--heuristic-name,--heuristic_name", HEURISTIC_NAME, "Name of the merge heuristic to use; default count_driven. Use any heuristic in the evaluation directory. It is often beneficial to write your own, as heuristics are very application specific.")->required();
    app.add_option("--data-name,--data_name", DATA_NAME, "Name of the merge data class to use; default count_data. Use any heuristic in the evaluation directory.");
    app.add_option("--evalpar", EVALUATION_PARAMETERS, "string of key-value pairs for evaluation functions");

    app.add_option("--satsolver", SAT_SOLVER, "Name of the SAT solver executable. Default=glucose.");
    app.add_option("--aptafile", APTA_FILE, "Name of the input file containing a previously learned automaton in json format. Note that you need to use the same evaluation function it was learned with.");
    app.add_option("--aptafile2", APTA_FILE2, "Name of the input file containing a previously learned automaton in json format. Note that you need to use the same evaluation function it was learned with.");

    app.add_option("--debug", DEBUGGING, "turn on debugging mode, printing includes pointers and find/union structure, more output");
    app.add_option("--addtails", ADD_TAILS, "Add tails to the states, used for splitting nodes. When not needed, it saves space and time to not add them. Default=1.");
    app.add_option("--random", RANDOMIZE_SCORES, "Amount of randomness r to include in merging heuristic. Each merge score s is modified to (s - s*random(0,r)). Default=0.");
    app.add_option("--runs", ENSEMBLE_RUNS, "Number of greedy runs/iterations; default=1. Advice: when using random greedy, a higher value is recommended (100 was used in Stamina winner).");
    app.add_option("--parentsizethreshold", PARENT_SIZE_THRESHOLD, "The minimum size for a node to be assigned children when reading inputs, subsequent events in longer traces are ignored. Useful for streaming and some evaluation functions. Default=-1.");
    app.add_option("--reversetraces", REVERSE_TRACES, "When set to true, flexfringe starts from a suffix instead of a prefix tree. Default = 0.");
    app.add_option("--slidingwindow", SLIDING_WINDOW, "Use a sliding window when reading CSV files. Default=0.");
    app.add_option("--swsize", SLIDING_WINDOW_SIZE, "The size of the sliding window if --slidingwindow is set to 1. Default=20.");
    app.add_option("--swstride", SLIDING_WINDOW_STRIDE, "The stride (jump size) between two sliding windows, when --slidingwindow is set to 1. Default=5.");
    app.add_option("--swtype", SLIDING_WINDOW_TYPE, "Whether the sliding window should use the last element as the sliding window type. Default = 0.");
    app.add_option("--swaddshorter", SLIDING_WINDOW_ADD_SHORTER, "Whether sliding windows shorter than swsize should be added to the apta. Default = 0.");
    app.add_option("--redbluethreshold", RED_BLUE_THRESHOLD, "Boolean. If set to 1, then states will only be appended to red- or blue states. Only makes sense in stream mode. Default=0.");

    app.add_option("--extend", EXTEND_ANY_RED, "When set to 1, any merge candidate (blue) that cannot be merged with any target (red) is immediately changed into a (red) target; default=1. If set to 0, a merge candidate is only changed into a target when no more merges are possible. Advice: unclear which strategy is best, when using statistical (or count-based) consistency checks, keep in mind that merge consistency between states may change due to other performed merges. This will especially influence low frequency states. When there are a lot of those, we therefore recommend setting x=0.");
    app.add_option("--shallowfirst", DEPTH_FIRST, "When set to 1, the ordering of the nodes is changed from most frequent first (default) to most shallow (smallest depth) first; default=0. Advice: use depth-first when learning from characteristic samples.");
    app.add_option("--largestblue", MERGE_MOST_VISITED, "When set to 1, the algorithm only tries to merge the most frequent (or most shallow if w=1) candidate (blue) states with any target (red) state, instead of all candidates; default=0. Advice: this reduces run-time significantly but comes with a potential decrease in merge quality");
    app.add_option("--blueblue", MERGE_BLUE_BLUE, "When set to 1, the algorithm tries to merge candidate (blue) states with candidate (blue) states in addition to candidate (blue) target (red) merges; default=0. Advice: this adds run-time to the merging process in exchange for potential improvement in merge quality.");
    app.add_option("--redfixed", RED_FIXED, "When set to 1, merges that add new transitions to red states are considered inconsistent. Merges with red states will also not modify any of the counts used in evaluation functions. Once a red state has been learned, it is considered final and unmodifiable; default=0. Advice: setting this to 1 frequently results in easier to visualize and more insightful models.");
    app.add_option("--allfixed", ALL_FIXED, "When set to 1, merges that add new transitions to any state are considered inconsistent. Merges with red states will also not modify any of the counts used in evaluation functions. Default=0. Advice: setting this to 1 leads to insightful but large models.");
    app.add_option("--ktail", KTAIL, "k-Tails (speedup parameter), only testing merges until depth k (although original ktails can produce non-deterministic machines, flexfringe cannot, it is purely for speedup). Default=-1");
    app.add_option("--origktail", IDENTICAL_KTAIL, "Original k-Tails, requires merged states to have identical suffixes (future paths) up to depth k. (although original ktails can produce non-deterministic machines, flexfringe cannot). Default=-1.");
    app.add_option("--kstate", KSTATE, "k-Tails for states (speedup parameter), only testing merges until states of size k. Default=-1.");
    app.add_option("--mergelocal", MERGE_LOCAL, "only perform local merges, up to APTA distance k, useful when learning from software data.");
    app.add_option("--mcollector", MERGE_LOCAL_COLLECTOR_COUNT, "when local merges are used, allow merges with non-local collector states, these are states with at least k input transitions.");
    app.add_option("--markovian", MARKOVIAN_MODEL, "learn a \"Markovian\" model that ensures the incoming transitions have the same label, resulting in a Markov graph (states correspond to a unique label, but the same label can occur in multiple places), any heuristic can be used. (default: 0)");
    app.add_option("--mergeroot", MERGE_ROOT, "Allow merges with the root? Default: 1 (true).");
    app.add_option("--testmerge", MERGE_WHEN_TESTING, "When set to 0, merge tries in order to compute the evaluation scores do not actually perform the merges themselves. Thus the consistency and score evaluation for states in merges that add recursive loops are uninfluenced by earlier merges; default=1. Advice: setting this to 0 reduces run-time and can be useful when learning models using statistical evaluation functions, but can lead to inconsistencies when learning from labeled data.");
    app.add_option("--mergedata", MERGE_DATA, "Whether to update data during the merging process (1) or keep the counts from the prefix tree intact (0). Default=1.");

    app.add_option("--sinkson", USE_SINKS, "Set to 1 to use sink states; default=1. Advice: leads to much more concise and easier to visualize models, but can cost predictive performance depending on the sink definitions.");
    app.add_option("--sinkcount", SINK_COUNT, "The maximum number of occurrences of a state for it to be a low count sink (see evaluation functions); default=10.");
    app.add_option("--mergesinks", MERGE_SINKS, "Whether to merge sinks with other sink nodes after the main merging process. default=0.");
    app.add_option("--mergesinkscore", MERGE_SINKS_WITH_CORE, "Whether to merge sinks with the red core (any other state) after the main merging process. default=0.");
    app.add_option("--satmergesinks", MERGE_SINKS_PRESOLVE, "Merge all sink nodes of the same type before sending the problem to the SAT solver (setting 0 or 1); default=1. Advice: radically improves runtime, only set to 0 when sinks of the same type can be different states in the final model.");
    app.add_option("--searchsinks", SEARCH_SINKS, "Start search process once all remaining blue states are sink nodes, use greedy before. Only valid for search strategies. Default 0 (false).");
    app.add_option("--sinkidentical", MERGE_IDENTICAL_SINKS, "Only merge sinks if they have identical suffixes. Default=0.");
    app.add_option("--convertsinks", CONVERT_SINK_STATES, "Instead of merging sinks, convert them to their form defined by the evaluation function (typically a garbage state). Default 0 (false).");
    app.add_option("--extendsinks", EXTEND_SINKS, "Only relevant when mergesinks is set to 1. When set to 1, sinks can be extended (aka, added to the core, colored red). When set to 0, all sinks will be merged with the current core. Default=1.");

    app.add_option("--finalprob", FINAL_PROBABILITIES, "model final probabilities? if set to 1, distributions are over Sigma*, otherwise over SigmaN. (default: 0)");
    app.add_option("--lowerbound", USE_LOWER_BOUND, "Does the merger use a minimum value of the heuristic function? Set using --lowerboundval. Default=0. Advice: state merging is forced to perform the merge with best heuristic value, it can sometimes be better to color a state red rather then performing a bad merge. This is achieved using a positive lower bound value. Models learned with positive lower bound are frequently more interpretable");
    app.add_option("--lowerboundval", LOWER_BOUND, "Minimum value of the heuristic function, smaller values are treated as inconsistent. Default=0. Advice: state merging is forced to perform the merge with best heuristic value, it can sometimes be better to color a state red rather then performing a bad merge. This is achieved using a positive lower bound value. Models learned with positive lower bound are frequently more interpretable");
    app.add_option("--extendscore", EXTEND_SCORE, "The score for an extend (not merge or split) refinement. Set this higher or equal to lowerboundval. Default=0.");
    app.add_option("--state_count", STATE_COUNT, "The minimum number of positive occurrences of a state for it to be included in overlap/statistical checks (see evaluation functions); default=25. Advice: low frequency states can have an undesired influence on statistical tests, set to at least 10. Note that different evaluation functions can use this parameter in different ways.");
    app.add_option("--symbol_count", SYMBOL_COUNT, "When set to 1, merge tries in order to compute the evaluation scores do not actually perform the merges themselves. Thus the consistency and score evaluation for states in merges that add recursive loops are uninfluenced by earlier merges; default=0. Advice: setting this to 1 reduces run-time and can be useful when learning models using statistical evaluation functions, but can lead to inconsistencies when learning from labeled data.");
    app.add_option("--correction", CORRECTION, "Value of a Laplace correction (smoothing) added to all symbol counts when computing statistical tests (in ALERGIA, LIKELIHOODRATIO, AIC, and KULLBACK-LEIBLER); default=0.0. Advice: unclear whether smoothing is needed for the different tests, more smoothing typically leads to smaller models.");
    app.add_option("--correction_seen", CORRECTION_SEEN, "Additional correction applied to seen values.");
    app.add_option("--correction_unseen", CORRECTION_UNSEEN, "Additional correction applied to unseen values.");
    app.add_option("--correction_per_seen", CORRECTION_PER_SEEN, "Additional correction, adds this correction to counts per seen value to both seen and unseen values.");
    app.add_option("--confidence_bound", CHECK_PARAMETER, "Extra parameter used during statistical tests, the significance level for the likelihood ratio test, the alpha value for ALERGIA; default=0.5. Advice: look up the statistical test performed, this parameter is not always the same as a p-value.");
    app.add_option("--typedist", TYPE_DISTRIBUTIONS, "Whether to perform tests on the type distributions of states. Default = 0.");
    app.add_option("--symboldist", SYMBOL_DISTRIBUTIONS, "Whether to perform tests on the symbol distributions of states. Default = 1.");
    app.add_option("--typeconsistent", TYPE_CONSISTENT, "Whether to enforce type consistency for states, i.e., to not merge positive states with negative ones. Default=1.");

    app.add_option("--satoffset", OFFSET, "DFASAT runs a SAT solver to find a solution of size at most the size of the partially learned DFA + E; default=5. Advice: larger values greatly increases run-time. Setting it to 0 is frequently sufficient (when the merge heuristic works well).");
    app.add_option("--satplus", EXTRA_STATES, "With every iteration, DFASAT tries to find solutions of size at most the best solution found + P, default=0. Advice: current setting only searches for better solutions. If a few extra states is OK, set it higher.");
    app.add_option("--satfinalred", TARGET_REJECTING, "Make all transitions from red states without any occurrences force to have 0 occurrences (similar to targeting a rejecting sink), (setting 0 or 1) before sending the problem to the SAT solver; default=0. Advice: the same as finalred but for the SAT solver. Setting it to 1 greatly improves solving speed.");
    app.add_option("--symmetry", SYMMETRY_BREAKING, "Add symmetry breaking predicates to the SAT encoding (setting 0 or 1), based on Ulyantsev et al. BFS symmetry breaking; default=1. Advice: in our experience this only improves solving speed.");
    app.add_option("--forcing", FORCING, "Add predicates to the SAT encoding that force transitions in the learned DFA to be used by input examples (setting 0 or 1); default=0. Advice: leads to non-complete models. When the data is sparse, this should be set to 1. It does make the instance larger and can have a negative effect on the solving time.");

    app.add_option("--printblue", PRINT_BLUE, "Print blue states in the .dot file? Default 0 (false).");
    app.add_option("--printwhite", PRINT_WHITE, "Print white states in the .dot file? These are typically sinks states, i.e., states that have not been considered for merging. Default 0 (false).");
    app.add_option("--outputsinks", OUTPUT_SINKS, "Print sink states and transition in a separate json file. Default 0 (false).");

    app.add_option("--depthcheck", PERFORM_DEPTH_CHECK, "In addition to standard state merging checks, perform a check layer-by-layer in the prefix tree. This is a try to get more information out of infrequent traces and aims to capture long-term dependencies. Default=0.");
    app.add_option("--symbolcheck", PERFORM_SYMBOL_CHECK, "In addition to standard state merging checks, perform a check symbol-by-symbol in the prefix tree. This is a try to get more information out of infrequent traces and aims to capture long-term dependencies. Default=0.");
    app.add_option("--depthcheckmaxdepth", DEPTH_CHECK_MAX_DEPTH, "In case of performing depth or symbol checks, this parameter gives the maximum depth to compute these tests for. Default=-1 (bounded by the prefix tree).");
    app.add_option("--mergecheck", PERFORM_MERGE_CHECK, "Perform the standard merge check from the core state-merging algorithm. When set to false, all merges evaluate to true except for other constraints such as locality, markovian, etc. Default=1.");

    app.add_option("--searchdeep", SEARCH_DEEP, "Search using a greedy call until no more merges can be performed. Default=0.");
    app.add_option("--searchlocal", SEARCH_LOCAL, "Search using the local heuristic from the evaluation function. Default=0.");
    app.add_option("--searchglobal", SEARCH_GLOBAL, "Search using the global heuristic from the evaluation function. Default=0.");
    app.add_option("--searchpartial", SEARCH_PARTIAL, "Search using the partial heuristic from the evaluation function. Default=0.");

    app.add_option("--predictreset", PREDICT_RESET, "When predicting and there is no outgoing transition, the model is reset to the root state. This works well when using sliding windows. Default=0.");
    app.add_option("--predictremain", PREDICT_REMAIN, "When predicting and there is no outgoing transition, the model is looped back into the current state. Default=0.");
    app.add_option("--predictalign", PREDICT_ALIGN, "When predicting and there is no outgoing transition, the model remaining trace is aligned to the model by jumping to any other state and skipping symbols. Default=0.");
    app.add_option("--predictminimum", PREDICT_MINIMUM, "Predict returns (or finds when aligning) the smallest probability of a symbol from a sequence. Default=0.");
    app.add_option("--predicttype", PREDICT_TYPE, "Predicting calls the predict type functions from the evaluation function. Default=0.");
    app.add_option("--predicttypepath", PREDICT_TYPE_PATH, "Predictings are made based on paths in addition to final states (if implemented by evaluation function). Default=0.");
    app.add_option("--predictsymbol", PREDICT_SYMBOL, "Predicting calls the predict symbol functions from the evaluation function. Default=0.");
    app.add_option("--predicttrace", PREDICT_TRACE, "Predicting calls the predict trace functions from the evaluation function. Default=1.");
    app.add_option("--predictdata", PREDICT_TRACE, "Predicting calls the predict data functions from the evaluation function. Default=0.");

    app.add_option("--aligndistancepenalty", ALIGN_DISTANCE_PENALTY, "A penalty for jumping during alignment multiplied by the merged prefix tree distance. Default: 0.0.");
    app.add_option("--alignskippenalty", ALIGN_SKIP_PENALTY, "A penalty for skipping during alignment. Default: 0.0.");

    app.add_option("--diffsize", DIFF_SIZE, "Behavioral differencing works by sampling diffsize traces and using these to compute KL-Divergence. Default=1000.");
    app.add_option("--diffmaxlength", DIFF_MAX_LENGTH, "The maximum length of traces sampled for differencing. Default=50.");
    app.add_option("--diffmin", DIFF_MIN, "The minimum score for the behavioral difference of a sampled trace. Default=-100.");

    // parameters specifically for CMS heuristic
    app.add_option("--numoftables", NROWS_SKETCHES, "Number of rows of sketches upon initialization.");
    app.add_option("--vectordimension", NCOLUMNS_SKETCHES, "Number of columns of sketches upon initialization.");
    app.add_option("--futuresteps", NSTEPS_SKETCHES, "Number of steps into future when storing future in sketches. Default: 2.");
    app.add_option("--conditionalprob", CONDITIONAL_PROB, "Do make the sketches conditional as strings. Default=false");
    app.add_option("--minhash", MINHASH, "Perform Min-Hash scheme on the ngrams. Only works in conjunction with --conditionalprob turned on. Default=false");
    app.add_option("--minhashsize", MINHASH_SIZE, "Perform Min-Hash scheme on the ngrams. Only works in conjunction with --conditionalprob turned on. Default=false");
    app.add_option("--alphabetsize", ALPHABET_SIZE, "An upper estimate on the alphabet size. Only needed with minhash-function turned on, in order to perform the permutation. Larger estimate increases runtime. Default=0");

    app.add_option("--streaming_batchsize", STREAMING_BATCH_SIZE, "Batchsize for streaming. Default=500");

    // mainly for space-save heuristic
    app.add_option("-e,--epsilon", EPSILON, "Epsilon parameter, determining approximation error.");
    app.add_option("-D,--delta", DELTA, "Delta param, the error rate.");
    app.add_option("--mu", MU, "Distinguishability parameter.");
    app.add_option("--pref_L", L, "The expected length of prefixes.");
    app.add_option("--pref_K", K, "Number of frequent items in sketches.");
    app.add_option("--bootstrap_R", R, "The number of bootstrapped examples.");
    
    // active learning parameters
    app.add_option("--active_learning_algorithm", ACTIVE_LEARNING_ALGORITHM, "The basic algorithm that runs through. Current options are (lstar). DEFAULT: lstar");
    app.add_option("--use_active_learning", DO_ACTIVE_LEARNING, "Perform active learning on top of the normal learner. 1 for true, 0 for false. Default: 0");
    app.add_option("--al_max_search_depth", AL_MAX_SEARCH_DEPTH, "The active learning search depth. Some uses can be disabled with a value <= 0. Critical in models where a maximum-string-length occurs, such as transformer-models. Default: 25");
    app.add_option("--al_num_cex_search", AL_NUM_CEX_PARAM, "Samples parameter indicating a number in the counterexample search. For example, in the random w-method the number of strings per node,\\
                                                                and in random string search it is the delay. Default: 5000");
    app.add_option("--al_oracle", AL_ORACLE, "The oracle that we're using. Has to be specified!");
    app.add_option("--al_oracle2", AL_ORACLE_2, "In case we are using a second oracle.");
    app.add_option("--al_system_under_learning", AL_SYSTEM_UNDER_LEARNING, "The system under learning. Has to be specified!");
    app.add_option("--al_system_under_learning_2", AL_SYSTEM_UNDER_LEARNING_2, "The second system under learning. Used if more than one type of oracle queries have  to be used, see e.g. Ldot/DAALDer.");
    app.add_option("--al_batch_size", AL_BATCH_SIZE, "The batch size as it is used in some of the active learning algorithms. Useful for example when querying networks, as it speeds up the inference on the networks. Trade-off is between speed and RAM size. Default: 256");
    app.add_option("--al_cex_search_strategy", AL_CEX_SEARCH_STRATEGY, "The strategy to search for counterexamples. Normally the random w-method will do, but depending on your case you might want to choose. Default: random_w_method.");
    app.add_option("--al_ii_handler_name", AL_II_NAME, "The name of the incomplete information instance. If left empty, then no ii handler is being created. Not all algorithms support ii handlers. Default: [empty].");

    // TODO: shall we delete the rejecting_label option?
    app.add_option("--al_rejecting_label", AL_REJECTING_LABEL, "The label as a string that is used for rejecting (non-accepting) behavior. Only in active learning mode. DEFAULT: 0");
    app.add_option("--al_start_symbol", AL_START_SYMBOL, "The <SOS> symbol (as per NLP convention) represented by an int value. A value of -1 means that it is unused. Only in active learning mode when querying networks. DEFAULT: -1");
    app.add_option("--al_end_symbol", AL_END_SYMBOL, "The <SOS> symbol (as per NLP convention) represented by an int value. A value of -1 means that it is unused. Only in active learning mode when querying networks. DEFAULT: -1");
    
    app.add_option("--postgresql-connstring", POSTGRESQL_CONNSTRING,
                   "The string that connects to a postgresql database. This is either a key value pairing or a URI. "
                   "https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-KEYWORD-VALUE You can omit this "
                   "and use environment variables instead. https://www.postgresql.org/docs/current/libpq-envars.html Set the "
                   "first positional argument to an empty string to learn from the SQL or provide a trace file to load the data.");
    app.add_option("--postgresql-tblname", POSTGRESQL_TBLNAME,
                   "The string that hold the name to the table with the traces. You must provide this to signal "
                   "learning from an SQL database. The database PostgreSQL connects to must contain this table and the "
                   "{POSTGRESQL_TBLNAME}_meta table.");
    app.add_option("--postgresql-droptbls", POSTGRESQL_DROPTBLS,
                   "With this option you can tell the program to drop the existing tables. Default=true");
    
    CLI11_PARSE(app, argc, argv)

    loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
    loguru::init(argc, argv);
    loguru::add_file(LOG_PATH.c_str(), loguru::Append, loguru::Verbosity_MAX);

    LOG_S(INFO) << "Starting flexfringe run";

    run();

    LOG_S(INFO) << "Ending flexfringe run normally";

    mem_store::erase();

    return 0;
}
#endif
