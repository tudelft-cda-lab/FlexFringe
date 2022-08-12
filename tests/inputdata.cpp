//
//#include "catch.hpp"
//#include "inputdata.h"
//#include <cstdio>
//
//using Catch::Matchers::Equals;
//
//TEST_CASE( "Input Data: strip header name whitespace", "[parsing]" ) {
//    // This tests makes sure that having spaces in front of the column names in the header
//    // does not break things, as it caused issues recognizing the special column types before
//    // (e.g. [timestamp, id, symb] vs [timestamp,id,symb])
//
//    std::string input_whitespace = "timestamp, id, symb\n"
//                                   "2022-08-04T11:00:14.707375+0200, 670edd27, Received symbol b\n"
//                                   "2022-08-04T11:00:14.707375+0200, 670edd27, Received symbol b\n"
//                                   "2022-08-04T11:00:14.707375+0200, 670edd28, Received symbol b";
//    std::istringstream input_stream_whitespace(input_whitespace);
//
//    std::string input_no_whitespace = "timestamp,id,symb\n"
//                                      "2022-08-04T11:00:14.707375+0200, 670edd27, Received symbol b\n"
//                                      "2022-08-04T11:00:14.707375+0200, 670edd27, Received symbol b\n"
//                                      "2022-08-04T11:00:14.707375+0200, 670edd28, Received symbol b";
//    std::istringstream input_stream_no_whitespace(input_no_whitespace);
//
//    inputdata input_data_whitespace;
//    input_data_whitespace.read_csv_header(input_stream_whitespace);
//    input_data_whitespace.read_csv_file(input_stream_whitespace);
//
//    inputdata input_data_no_whitespace;
//    input_data_no_whitespace.read_csv_header(input_stream_no_whitespace);
//    input_data_no_whitespace.read_csv_file(input_stream_no_whitespace);
//
//    auto trace_w_it = input_data_whitespace.traces_start();
//    auto trace_nw_it = input_data_no_whitespace.traces_start();
//    auto trace_w_end = input_data_whitespace.traces_end();
//    auto trace_nw_end = input_data_no_whitespace.traces_end();
//
//    while (trace_w_it != trace_w_end && trace_nw_it != trace_nw_end) {
//        auto trace_w = *trace_w_it++;
//        auto trace_nw = *trace_nw_it++;
//        REQUIRE_THAT(trace_w->to_string(), Equals(trace_nw->to_string()));
//    }
//}
//
//
//TEST_CASE( "Input Data: read symbol column with whitespace", "[parsing]" ) {
//    // There was an issue where if the symbol column included whitespace, only the part up until
//    // the first whitespace would be stored, e.g. "Received symbol a" -> "Received"
//
//    std::string inputstring = "timestamp, id, symb\n"
//                              "2022-08-04T11:00:14.707375+0200, 670edd27, Received symbol a\n"
//                              "2022-08-04T11:00:14.707375+0200, 670edd28, Received symbol b\n"
//                              "2022-08-04T11:00:14.707375+0200, 670edd29, Received symbol c";
//    std::istringstream input_stream_whitespace(inputstring);
//
//    inputdata input_data;
//    input_data.read_csv_header(input_stream_whitespace);
//    input_data.read_csv_file(input_stream_whitespace);
//
//    std::vector<string> expected_symbols = {
//            "Received symbol a",
//            "Received symbol b",
//            "Received symbol c"
//    };
//
//    for (auto trace = input_data.traces_start();
//         trace != input_data.traces_end();
//         ++trace) {
//        auto cur_trace = *trace;
//
//        string cur_symbol = cur_trace->get_head()->to_string();
//        string cur_expected = expected_symbols.back();
//        expected_symbols.pop_back();
//
//        REQUIRE_THAT(cur_symbol, Equals(cur_expected));
//    }
//}
