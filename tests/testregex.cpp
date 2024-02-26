#include "catch.hpp"
#include "loguru.hpp"
#include "parameters.h"

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



TEST_CASE( "Test regex_builder functionality", "[regex]" ) {
    HEURISTIC_NAME = "evidence_driven";
    DATA_NAME = "edsm_data";
    APTA_FILE = "tests/data/regex/simple1.final.json";
    INPUT_FILE = "tests/data/regex/simple.txt.dat";
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
    std::map<int, std::regex> regexes = {};
    for (const int type : inputdata_locator::get()->get_types()) {
        regexes[type] = std::regex{ builder.to_regex(type) };
    }

    std::ifstream input_stream{ INPUT_FILE };
    auto input_parser = abbadingoparser(input_stream);
    auto parser_strategy = in_order();

    for (auto* trace : id.trace_iterator(input_parser, parser_strategy)) {
        const std::string tr_str = sqldb::vec2str(trace->get_input_sequence(false, false));
        bool match = std::regex_match(tr_str, regexes[trace->type]);
        CHECK( match );
    }
}

