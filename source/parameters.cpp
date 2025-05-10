#include "parameters.h"
#include <random>

std::uniform_real_distribution<double> unif(0.0, 1.0);
std::default_random_engine re;
double random_double(){ return unif(re); }

std::string HEURISTIC_NAME = "alergia";
std::string DATA_NAME = "default";
std::string EVALUATION_PARAMETERS = "";
std::string INPUT_FILE = "test.dat";
std::string OUTPUT_FILE = "";
std::string OUTPUT_TYPE = "both";
std::string LOG_PATH = "flexfringe.log";

std::string OPERATION_MODE = "greedy";

std::string SAT_SOLVER = "";
std::string APTA_FILE = "";
std::string APTA_FILE2 = "";

std::string COMMAND_LINE = "";

bool DEBUGGING = false;
bool ADD_TAILS = true;
bool RED_BLUE_THRESHOLD = false;
double RANDOMIZE_SCORES = 0.0;
int ENSEMBLE_RUNS = 1;
int PARENT_SIZE_THRESHOLD = -1;
bool REVERSE_TRACES = false;
bool SLIDING_WINDOW = false;
int SLIDING_WINDOW_SIZE = 20;
int SLIDING_WINDOW_STRIDE = 5;
bool SLIDING_WINDOW_TYPE = false;
bool SLIDING_WINDOW_ADD_SHORTER = true;
bool STORE_ACCESS_STRINGS = true;

bool EXTEND_ANY_RED = 0;
bool DEPTH_FIRST = 0;
bool MERGE_MOST_VISITED = 0;
bool MERGE_BLUE_BLUE = 0;
bool RED_FIXED = 0;
bool ALL_FIXED = 0;
int KTAIL = -1;
int IDENTICAL_KTAIL = -1;
int KSTATE = -1;
int MERGE_LOCAL = -1;
int MERGE_LOCAL_COLLECTOR_COUNT = -1;
int MARKOVIAN_MODEL = 0;
bool MERGE_ROOT = true;
bool MERGE_WHEN_TESTING = true;
bool MERGE_DATA = true;
bool STAR_FREE = false;

bool USE_SINKS = 0;
int SINK_COUNT = 10;
bool MERGE_SINKS = 0;
bool MERGE_SINKS_WITH_CORE = 0;
bool MERGE_SINKS_PRESOLVE = 0;
bool SEARCH_SINKS = false;
bool MERGE_IDENTICAL_SINKS = false;
bool CONVERT_SINK_STATES = false;
bool EXTEND_SINKS = true;
bool SINK_TYPE = false;

bool FINAL_PROBABILITIES = 0;
bool USE_LOWER_BOUND = 0;
float LOWER_BOUND = 0;
double EXTEND_SCORE = 0.0;
int STATE_COUNT = 25;
int SYMBOL_COUNT = 10;
float CORRECTION = 1.0;
double CORRECTION_SEEN = 0.0;
double CORRECTION_UNSEEN = 0.0;
double CORRECTION_PER_SEEN = 0.0;
float CHECK_PARAMETER = 0.05;
bool TYPE_DISTRIBUTIONS = false;
bool SYMBOL_DISTRIBUTIONS = true;
bool TYPE_CONSISTENT = true;

int OFFSET = 1;
int EXTRA_STATES = 0;
bool TARGET_REJECTING = 0;
bool SYMMETRY_BREAKING = 0;
bool FORCING = 0;

bool PRINT_WHITE = false;
bool PRINT_BLUE = false;
bool PRINT_RED = true;
bool OUTPUT_SINKS = false;

bool PERFORM_SYMBOL_CHECK = false;
bool PERFORM_DEPTH_CHECK = false;
int DEPTH_CHECK_MAX_DEPTH = -1;
bool PERFORM_MERGE_CHECK = true;

bool SEARCH_DEEP = false;
bool SEARCH_LOCAL = false;
bool SEARCH_GLOBAL = false;
bool SEARCH_PARTIAL = false;

bool PREDICT_RESET = false;
bool PREDICT_REMAIN = false;
bool PREDICT_ALIGN = false;
bool PREDICT_MINIMUM = false;
bool PREDICT_TYPE = false;
bool PREDICT_TYPE_PATH = false;
bool PREDICT_SYMBOL = false;
bool PREDICT_TRACE = true;
bool PREDICT_DATA = false;

// the count-min-sketches
int NROWS_SKETCHES = 0;
int NCOLUMNS_SKETCHES = 0;
int DISTANCE_METRIC_SKETCHES = 1;
int RANDOM_INITIALIZATION_SKETCHES = 0;
int NSTEPS_SKETCHES = 2;

double ALIGN_SKIP_PENALTY = 1.0;
double ALIGN_DISTANCE_PENALTY = 0.1;

int DIFF_SIZE = 1000;
int DIFF_MAX_LENGTH = 50;
double DIFF_MIN = -100.0;



