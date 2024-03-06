
#include "catch.hpp"

#include "evaluate.h"
#include "greedy.h"
#include "evaluation_factory.h"
#include "parameters.h"
#include "input/inputdata.h"
#include "input/inputdatalocator.h"
#include "input/parsers/abbadingoparser.h"

using Catch::Matchers::Equals;

//TODO: refactor: These should probably be taken out of main.cpp
evaluation_function* get_evaluation();
void print_current_automaton(state_merger*, const std::string&, const std::string&);

TEST_CASE( "Smoke test: greedy alergia on stamina 1_training", "[smoke]" ) {
    HEURISTIC_NAME = "alergia";
    DATA_NAME = "alergia_data";

    evaluation_function *eval = get_evaluation();
    REQUIRE(eval != nullptr);

    std::ifstream input_stream("data/staminadata/1_training.txt");
    if (!input_stream) {
        std::cerr << "Error: " << strerror(errno);
    }
    REQUIRE(input_stream);

    inputdata id;
    inputdata_locator::provide(&id);
    auto parser = abbadingoparser(input_stream);
    id.read(&parser);

    apta the_apta;
    state_merger merger(&id, eval, &the_apta);
    the_apta.set_context(&merger);
    eval->set_context(&merger);

    eval->initialize_before_adding_traces();
    id.add_traces_to_apta(&the_apta);
    eval->initialize_after_adding_traces(&merger);

    // Double check all apta nodes have an access trace
    for(APTA_iterator Ait = APTA_iterator(the_apta.get_root()); *Ait != 0; ++Ait) {
        apta_node *n = *Ait;
        assert(n->get_access_trace() != nullptr);
    }

    greedy_run(&merger);

    // print_current_automaton(&merger, "/tmp/flexfringe_test_out", ".final");

    //TODO: verify learned state machine is reasonable

//    delete merger;
}

TEST_CASE( "Smoke test: greedy edsm on stamina 1_training", "[smoke]" ) {
    HEURISTIC_NAME = "evidence_driven";
    DATA_NAME = "edsm_data";

    evaluation_function *eval = get_evaluation();
    REQUIRE(eval != nullptr);

    std::ifstream input_stream("data/staminadata/1_training.txt");
    REQUIRE(input_stream);

    inputdata id;
    inputdata_locator::provide(&id);
    auto parser = abbadingoparser(input_stream);
    id.read(&parser);

    apta* the_apta = new apta();
    auto* merger = new state_merger(&id, eval, the_apta);
    the_apta->set_context(merger);
    eval->set_context(merger);

    eval->initialize_before_adding_traces();
    id.add_traces_to_apta(the_apta);
    eval->initialize_after_adding_traces(merger);

    greedy_run(merger);

    //print_current_automaton(merger, "/tmp/flexfringe_test_out", ".final");

    //TODO: verify learned state machine is reasonable

//    delete merger;
}

TEST_CASE( "Smoke test: abbadingo input data with empty traces", "[smoke]" ) {
    std::ifstream input_stream("data/PAutomaC-competition_sets/1.pautomac.train.dat");
    if (!input_stream) {
        std::cerr << "Error: " << strerror(errno);
    }
    REQUIRE(input_stream);

    inputdata id;
    inputdata_locator::provide(&id);
    auto parser = abbadingoparser(input_stream);
    id.read(&parser);

    REQUIRE(id.get_num_sequences() == 20000);
}

// This tests verifies that the dot file output works as expected
// It will need updating whenever the dot output changes
// TODO: Figure out a way to check if the dot output is valid without running graphviz
TEST_CASE( "Smoke test: dot output", "[smoke]" ) {
    HEURISTIC_NAME = "evidence_driven";
    DATA_NAME = "edsm_data";

    evaluation_function *eval = get_evaluation();
    REQUIRE(eval != nullptr);

    std::string input = "12 4\n"
                        "1 3 a b c\n"
                        "1 3 a b d\n"
                        "0 2 a b\n"
                        "0 2 a a\n"
                        "0 2 b b\n"
                        "0 1 c\n"
                        "0 2 c c\n"
                        "0 1 d\n"
                        "0 2 d d\n"
                        "0 1 a\n"
                        "0 2 b c\n"
                        "0 2 b d\n";
    std::istringstream input_stream(input);

    inputdata id;
    inputdata_locator::provide(&id);
    auto parser = abbadingoparser(input_stream);
    id.read(&parser);

    apta* the_apta = new apta();
    auto* merger = new state_merger(&id, eval, the_apta);
    the_apta->set_context(merger);
    eval->set_context(merger);

    eval->initialize_before_adding_traces();
    id.add_traces_to_apta(the_apta);
    eval->initialize_after_adding_traces(merger);

    greedy_run(merger);

    std::stringstream dot_stream;
    merger->print_dot(dot_stream);
    std::string actual_output = dot_stream.str();

    std::string expected_output = "// produced with flexfringe // \n"
                                  "digraph DFA {\n"
                                  "\t-1 [label=\"root\" shape=box];\n"
                                  "\t\tI -> -1;\n"
                                  "\t-1 [ label=\"-1 #25 fin: 0:8 , \n"
                                  " path: 1:2 , 0:15 , \" , style=filled, fillcolor=\"firebrick1\", width=1.44882, height=1.44882, penwidth=3.2581];\n"
                                  "\t\t-1 -> 1 [label=\"a \" , penwidth=3.2581 ];\n"
                                  "\t\t-1 -> -1 [label=\"b \" , penwidth=3.2581 ];\n"
                                  "\t\t-1 -> -1 [label=\"c \" , penwidth=3.2581 ];\n"
                                  "\t\t-1 -> -1 [label=\"d \" , penwidth=3.2581 ];\n"
                                  "\t1 [ label=\"1 #8 fin: 0:2 , \n"
                                  " path: 1:4 , 0:2 , \" , style=filled, fillcolor=\"firebrick1\", width=1.16228, height=1.16228, penwidth=2.19722];\n"
                                  "\t\t1 -> -1 [label=\"a \" , penwidth=2.19722 ];\n"
                                  "\t\t1 -> 1 [label=\"b \" , penwidth=2.19722 ];\n"
                                  "\t\t1 -> 3 [label=\"c \" , penwidth=2.19722 ];\n"
                                  "\t\t1 -> 3 [label=\"d \" , penwidth=2.19722 ];\n"
                                  "\t3 [ label=\"3 #2 fin: 1:2 , \n"
                                  " path: \" , style=filled, fillcolor=\"firebrick1\", width=0.741276, height=0.741276, penwidth=1.09861];\n"
                                  "}\n";

    REQUIRE_THAT(actual_output, Equals(expected_output));
}