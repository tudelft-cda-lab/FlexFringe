#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include <string>
#include <vector>

using namespace std;

double random_double();

extern bool MERGE_SINKS;
extern int STATE_COUNT;
extern int SYMBOL_COUNT;
extern int SINK_COUNT;
extern float CORRECTION;
extern float CHECK_PARAMETER;
extern bool USE_SINKS;
extern bool USE_LOWER_BOUND;
extern float LOWER_BOUND;
extern int alphabet_size;
extern bool EXTEND_ANY_RED;
extern bool MERGE_SINKS_PRESOLVE;
extern int OFFSET;
extern int EXTRA_STATES;
extern bool TARGET_REJECTING;
extern bool SYMMETRY_BREAKING;
extern bool FORCING;
extern string OUTPUT_TYPE;
extern string LOG_PATH;
extern string DEBUG_DIR;
extern bool MERGE_MOST_VISITED;
extern bool MERGE_BLUE_BLUE;
extern bool RED_FIXED;
extern bool ALL_FIXED;
extern bool MERGE_WHEN_TESTING;
extern bool DEPTH_FIRST;
extern int RANGE;
extern string COMMAND;
extern bool FINAL_PROBABILITIES;
extern int MERGE_LOCAL;
extern int MERGE_LOCAL_COLLECTOR_COUNT;
extern int KTAIL;
extern int KSTATE;
extern int MARKOVIAN_MODEL;
extern bool MERGE_SINKS_WITH_CORE;

extern bool TYPE_DISTRIBUTIONS;
extern bool SYMBOL_DISTRIBUTIONS;
extern bool TYPE_CONSISTENT;

extern bool MERGE_ROOT;
extern bool PRINT_WHITE;
extern bool PRINT_BLUE;
extern bool PRINT_RED;

extern double EXTEND_SCORE;

extern bool SEARCH_SINKS;
extern bool CONVERT_SINK_STATES;
extern bool EXTEND_SINKS;
extern bool MERGE_DATA;

extern bool PREDICT_RESET;
extern bool PREDICT_REMAIN;
extern bool PREDICT_ALIGN;
extern bool PREDICT_MINIMUM;

extern double RANDOMIZE_SCORES;

extern bool ADD_TAILS;
extern int PARENT_SIZE_THRESHOLD;
extern bool RED_BLUE_THRESHOLD;

extern bool PERFORM_DEPTH_CHECK;
extern int DEPTH_CHECK_MAX_DEPTH;

extern bool PERFORM_MERGE_CHECK;

extern bool MERGE_IDENTICAL_SINKS;

extern bool OUTPUT_SINKS;

extern bool SLIDING_WINDOW;
extern int SLIDING_WINDOW_SIZE;
extern int SLIDING_WINDOW_STRIDE;
extern bool SLIDING_WINDOW_TYPE;

extern bool SEARCH_DEEP;
extern bool SEARCH_LOCAL;
extern bool SEARCH_GLOBAL;
extern bool SEARCH_PARTIAL;

extern bool REVERSE_TRACES;

extern double CORRECTION_SEEN;
extern double CORRECTION_UNSEEN;
extern double CORRECTION_PER_SEEN;

extern bool PREDICT_TYPE;
extern bool PREDICT_TYPE_PATH;
extern bool PREDICT_SYMBOL;
extern bool PREDICT_TRACE;

extern double ALIGN_SKIP_PENALTY;
extern double ALIGN_DISTANCE_PENALTY;

extern bool SLIDING_WINDOW_ADD_SHORTER;

extern bool PREDICT_DATA;

extern int DIFF_SIZE;
extern double DIFF_MIN;
extern int DIFF_MAX_LENGTH;

extern bool STORE_ACCESS_STRINGS;

extern int BATCH_SIZE;

// Count-min-sketches
extern int NROWS_SKETCHES;
extern int NCOLUMNS_SKETCHES;
extern int NSTEPS_SKETCHES;
extern bool CONDITIONAL_PROB;
extern bool MINHASH;
extern int MINHASH_SIZE;
extern int ALPHABET_SIZE;

// space-saving specific
extern double DELTA;
extern double MU;
extern double EPSILON;
extern int L;
extern int R;
extern int K;

extern string HEURISTIC_NAME;
extern string DATA_NAME;
extern string INPUT_FILE;
extern string OUTPUT_FILE;
extern string EVALUATION_PARAMETERS;
extern string OPERATION_MODE;

extern string SAT_SOLVER;
extern string APTA_FILE;
extern string APTA_FILE2;

extern string COMMAND_LINE;

extern bool DEBUGGING;
extern int ENSEMBLE_RUNS;
extern bool PERFORM_SYMBOL_CHECK;
extern int IDENTICAL_KTAIL;

extern bool STAR_FREE;
extern bool SINK_TYPE;

// active learning parameters
extern string ACTIVE_LEARNING_ALGORITHM;
extern bool DO_ACTIVE_LEARNING;
extern string REJECTING_LABEL;

extern int START_SYMBOL;
extern int END_SYMBOL;

extern string POSTGRESQL_CONNSTRING;
extern string POSTGRESQL_TBLNAME;
extern bool POSTGRESQL_DROPTBLS;

#endif
