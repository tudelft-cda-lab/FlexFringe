/**
 * @file active_learning_main.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The main subroutine that is starting the active learning.
 * @version 0.1
 * @date 2023-03-08
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "active_learning_main.h"

#include "algorithm_base.h"
#include "ldot.h"
#include "lsharp.h"
#include "lstar.h"
#include "lstar_imat.h"
#include "probabilistic_lsharp.h"
#include "transformer_l_sharp.h"
#include "transformer_weighted_lsharp.h"
#include "weighted_lsharp.h"
#include "paul.h"

#include "active_sul_oracle.h"
#include "dfa_sul.h"
#include "input_file_oracle.h"
#include "input_file_sul.h"
#include "nn_sul_base.h"
#include "nn_weighted_output_sul.h"
#include "sqldb_sul.h"
#include "sqldb_sul_random_oracle.h"
#include "sqldb_sul_regex_oracle.h"

#include "base_teacher.h"
#include "teacher_imat.h"

#include "sqldb_sul_regex_oracle.h"

#include "abbadingoparser.h"
#include "csvparser.h"
#include "input/abbadingoreader.h"
#include "inputdata.h"
#include "inputdatalocator.h"
#include "main_helpers.h"
#include "parameters.h"

#include "misc/printutil.h"
#include "utility/loguru.hpp"

#include <cassert>
#include <fstream>
#include <stdexcept>
#include <string>

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace std;
using namespace active_learning_namespace;

ifstream active_learning_main_func::get_inputstream() const {
    ifstream input_stream(INPUT_FILE);
    cout << "Input file: " << INPUT_FILE << endl;
    if (!input_stream) {
        cerr << "Input file not found, aborting" << endl;
        exit(-1);
    } else {
        cout << "Using input file: " << INPUT_FILE << endl;
    }
    return input_stream;
}

std::unique_ptr<parser> active_learning_main_func::get_parser(ifstream& input_stream) const {
    if (INPUT_FILE.ends_with(".csv")) {
        return make_unique<csv_parser>(input_stream, csv::CSVFormat().trim({' '}));
    } else {
        return make_unique<abbadingoparser>(input_stream);
    }
}

inputdata* active_learning_main_func::get_inputdata() const {
    ifstream input_stream = get_inputstream();

    inputdata* id = inputdata_locator::get();

    auto input_parser = get_parser(input_stream);
    id->read(input_parser.get());
    input_stream.close();
    return id;
}

/**
 * @brief Selects the SUL to be used.
 *
 * @return shared_ptr<sul_base> The sul.
 */
shared_ptr<sul_base> active_learning_main_func::select_sul_class(const bool ACTIVE_SUL) const {
    if (ACTIVE_SUL) {
        // TODO: select the SUL better than you do here
        if (SQLDB) {
            return shared_ptr<sul_base>(new sqldb_sul(*my_sqldb));
        }

        string CONNECTOR_SCRIPT = APTA_FILE2.size() == 0 ? INPUT_FILE : APTA_FILE2;
        if (CONNECTOR_SCRIPT.compare(CONNECTOR_SCRIPT.length() - 3, CONNECTOR_SCRIPT.length(), ".py") == 0) {
            return shared_ptr<sul_base>(new nn_weighted_output_sul(CONNECTOR_SCRIPT));
        }

        return shared_ptr<sul_base>(new dfa_sul());
    }
    return shared_ptr<sul_base>(new input_file_sul());
}

/**
 * @brief Selects the teacher to be used. In case alternative teachers want to be written.
 *
 * @return unique_ptr<base_teacher> The teacher.
 */
std::unique_ptr<base_teacher> active_learning_main_func::select_teacher_class(shared_ptr<sul_base>& sul,
                                                                         const bool ACTIVE_SUL) const {
    if (ACTIVE_LEARNING_ALGORITHM == "l_star_imat") {
        return std::unique_ptr<base_teacher>(new teacher_imat(sul));
    }
    return std::unique_ptr<base_teacher>(new base_teacher(sul));
}

/**
 * @brief Selects the oracle to be used. In case alternative oracles want to be written.
 *
 * @return unique_ptr<eq_oracle_base> The oracle.
 */
std::unique_ptr<eq_oracle_base> active_learning_main_func::select_oracle_class(shared_ptr<sul_base>& sul,
                                                                          const bool ACTIVE_SUL) const {
    if (ACTIVE_LEARNING_ALGORITHM == "l_star_imat") {
        return std::unique_ptr<eq_oracle_base>(new sqldb_sul_random_oracle(sul));
    }
    if (SQLDB) {
        return std::unique_ptr<eq_oracle_base>(new sqldb_sul_regex_oracle(sul));
    }
    if (ACTIVE_SUL) {
        return std::unique_ptr<eq_oracle_base>(new active_sul_oracle(sul));
    }
    return std::unique_ptr<eq_oracle_base>(new input_file_oracle(sul));
}

/**
 * @brief Selects the parameters the algorithm runs with and runs the algorithm.
 *
 */
void active_learning_main_func::run_active_learning() {
    inputdata id;
    inputdata_locator::provide(&id);
    assertm(ENSEMBLE_RUNS > 0, "nruns parameter must be larger than 0 for active learning.");

    // Setting some initialization for learning from SQLDB.
    SQLDB = POSTGRESQL_TBLNAME != "";
    bool LOADSQLDB = INPUT_FILE != "";

    if (SQLDB) {
        if (!LOADSQLDB) {
            // If reading, not loading, from db, do not drop on initialization.
            POSTGRESQL_DROPTBLS = false;
        }
        my_sqldb = make_unique<psql::db>(POSTGRESQL_TBLNAME, POSTGRESQL_CONNSTRING);
        if (LOADSQLDB) {
            LOG_S(INFO) << "Loading from trace file " + INPUT_FILE;
            ifstream input_stream = get_inputstream();
            abbadingo_inputdata id;
            inputdata_locator::provide(&id);
            /* auto input_parser = get_parser(input_stream); */
            my_sqldb->load_traces(id, input_stream);
            LOG_S(INFO) << "Traces loaded.";
            return;
        }
    }

    const bool ACTIVE_SUL = !APTA_FILE.empty() || SQLDB ||
                            (INPUT_FILE.compare(INPUT_FILE.length() - 5, INPUT_FILE.length(), ".json") == 0) ||
                            (INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".dot") == 0) ||
                            (INPUT_FILE.compare(INPUT_FILE.length() - 3, INPUT_FILE.length(), ".py") == 0);

    auto sul = select_sul_class(ACTIVE_SUL);
    auto teacher = select_teacher_class(sul, ACTIVE_SUL);
    auto oracle = select_oracle_class(sul, ACTIVE_SUL);

    std::unique_ptr<algorithm_base> algorithm;
    if (ACTIVE_LEARNING_ALGORITHM == "l_star") {
        algorithm = std::unique_ptr<algorithm_base>(new lstar_algorithm(sul, teacher, oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "l_star_imat") {
        STORE_ACCESS_STRINGS = true;
        algorithm = std::unique_ptr<algorithm_base>(new lstar_imat_algorithm(sul, teacher, oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "l_sharp") {
        STORE_ACCESS_STRINGS = true;
        algorithm = std::unique_ptr<algorithm_base>(new lsharp_algorithm(sul, teacher, oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "p_l_sharp") {
        STORE_ACCESS_STRINGS = true;
        algorithm = std::unique_ptr<algorithm_base>(new probabilistic_lsharp_algorithm(sul, teacher, oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "weighted_l_sharp") {
        STORE_ACCESS_STRINGS = true;
        algorithm = std::unique_ptr<algorithm_base>(new weighted_lsharp_algorithm(sul, teacher, oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "l_dot") {
        STORE_ACCESS_STRINGS = true; // refinement uses this to get nodes, but that seems buggy somehow.
        algorithm = std::unique_ptr<algorithm_base>(new ldot_algorithm(sul, teacher, oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "transformer_l_sharp") {
        STORE_ACCESS_STRINGS = true;
        algorithm = std::unique_ptr<algorithm_base>(new transformer_lsharp_algorithm(sul, teacher, oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "weighted_transformer_l_sharp") {
        STORE_ACCESS_STRINGS = true;
        algorithm = std::unique_ptr<algorithm_base>(new transformer_weighted_lsharp_algorithm(sul, teacher, oracle));
    } else if (ACTIVE_LEARNING_ALGORITHM == "paul") {
        STORE_ACCESS_STRINGS = true;
        algorithm = std::unique_ptr<algorithm_base>(new paul_algorithm(sul, teacher, oracle));
    } else {
        throw logic_error("Fatal error: Unknown active_learning_algorithm flag used: " + ACTIVE_LEARNING_ALGORITHM);
    }

    if (ACTIVE_SUL && ACTIVE_LEARNING_ALGORITHM != "paul") {
        LOG_S(INFO) << "We do not want to run the input file, alphabet and input data must be inferred from SUL.";

        sul->pre(id);
        algorithm->run(id);
    } else {
        LOG_S(INFO) << "Learning (partly) passively. Therefore read in input-data.";
        get_inputdata();

        sul->pre(id);
        algorithm->run(id);
    }
}
