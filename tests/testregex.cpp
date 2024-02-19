#undef _GLIBCXX_REGEX_STATE_LIMIT 
#define _GLIBCXX_REGEX_STATE_LIMIT 100000000000000000

#include "catch.hpp"
#include "loguru.hpp"
#include "parameters.h"
#include <boost/regex.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <regex>
#include "regex_builder.h"
#include "misc/sqldb.h"
#include "input/inputdata.h"
#include "input/inputdatalocator.h"
#include "input/parsers/reader_strategy.h"
#include "input/parsers/abbadingoparser.h"
#include "state_merger.h"
#include "main_helpers.h"
#include "misc/printutil.h"



TEST_CASE( "Test regex_builder functionality", "[regex]" ) {
    HEURISTIC_NAME = "evidence_driven";
    DATA_NAME = "edsm_data";
    APTA_FILE = "data/tester1.final.json";
    INPUT_FILE = "data/staminadata/1_training.txt";
    PREDICT_TYPE = true;

    inputdata id;
    inputdata_locator::provide(&id);
    apta the_apta = apta();
    evaluation_function *eval = get_evaluation();
    state_merger merger = state_merger(&id, eval, &the_apta);


    std::ifstream input_apta_stream{ APTA_FILE };
    LOG_S(INFO) << "Reading apta file - " << APTA_FILE;
    the_apta.read_json(input_apta_stream);
    LOG_S(INFO) << "Finished reading apta file.";
    auto coloring = std::make_tuple(PRINT_RED, PRINT_BLUE, PRINT_WHITE);
    regex_builder builder = regex_builder(the_apta, merger, coloring, sqldb::num2str);
    LOG_S(INFO) << "Finished building the regex builder";
    std::map<int, boost::regex> regexes = {};
    for (const int type : inputdata_locator::get()->get_types()) {
        regexes[type] = boost::regex{ "^(" + builder.to_regex(type) + ")$" };
    }

    std::ifstream input_stream{ INPUT_FILE };
    auto input_parser = abbadingoparser(input_stream);
    auto parser_strategy = in_order();

    for (auto* trace : id.trace_iterator(input_parser, parser_strategy)) {
        const std::string tr_str = sqldb::vec2str(trace->get_input_sequence(false, false));
        bool match = boost::regex_match(tr_str, regexes[trace->type]);
        cout << trace->type << " " << tr_str << " " << match << endl;
    }
}

