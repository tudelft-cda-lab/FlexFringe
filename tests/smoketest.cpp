
#include "catch.hpp"

#include "evaluate.h"
#include "greedy.h"
#include "evaluation_factory.h"
#include "parameters.h"
#include "input/inputdata.h"
#include "input/inputdatalocator.h"
#include "input/parsers/abbadingoparser.h"

//TODO: refactor: These should probably be taken out of main.cpp
evaluation_function* get_evaluation();
void print_current_automaton(state_merger*, const string&, const string&);

TEST_CASE( "Smoke test: greedy alergia on stamina 1_training", "[smoke]" ) {
    HEURISTIC_NAME = "alergia";
    DATA_NAME = "alergia_data";

    evaluation_function *eval = get_evaluation();
    REQUIRE(eval != nullptr);

    ifstream input_stream("data/staminadata/1_training.txt");
    if (!input_stream) {
        cerr << "Error: " << strerror(errno);
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

    ifstream input_stream("data/staminadata/1_training.txt");
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
    HEURISTIC_NAME = "alergia";
    DATA_NAME = "alergia_data";

    evaluation_function *eval = get_evaluation();
    REQUIRE(eval != nullptr);

    ifstream input_stream("data/PAutomaC-competition_sets/1.pautomac.train.dat");
    if (!input_stream) {
        cerr << "Error: " << strerror(errno);
    }
    REQUIRE(input_stream);

    inputdata id;
    inputdata_locator::provide(&id);
    auto parser = abbadingoparser(input_stream);
    id.read(&parser);

    REQUIRE(id.get_num_sequences() == 20000);
}