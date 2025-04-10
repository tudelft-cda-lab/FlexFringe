include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithms"
        "${CMAKE_CURRENT_SOURCE_DIR}/databases"
        "${CMAKE_CURRENT_SOURCE_DIR}/oracle"
        "${CMAKE_CURRENT_SOURCE_DIR}/oracle/cex_search_strategies"
        "${CMAKE_CURRENT_SOURCE_DIR}/oracle/cex_conflict_search"
        "${CMAKE_CURRENT_SOURCE_DIR}/oracle/conflict_detectors"
        "${CMAKE_CURRENT_SOURCE_DIR}/active_learning_util"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory/incomplete_information"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory/incomplete_information/initializers"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory/incomplete_information/initializers/distinguishing_sequences"
        "${CMAKE_CURRENT_SOURCE_DIR}/system_under_learning"
        "${CMAKE_CURRENT_SOURCE_DIR}/system_under_learning/neural_network_suls"
        "${CMAKE_CURRENT_SOURCE_DIR}/system_under_learning/benchmark_parsers"
        )


add_library(ActiveLearning STATIC
            active_learning_mode.h
            active_learning_mode.cpp

            active_learning_util/definitions.h
            active_learning_util/definitions.cpp
            active_learning_util/common_functions.h
            active_learning_util/common_functions.cpp
            active_learning_util/random_int_generator.h
            active_learning_util/random_int_generator.cpp
            active_learning_util/factories.h
            active_learning_util/factories.cpp

            algorithms/algorithm_base.h
            algorithms/algorithm_base.cpp

            # DFAs
            algorithms/lstar.h
            algorithms/lstar.cpp
            algorithms/lstar_imat.h
            algorithms/lstar_imat.cpp
            algorithms/lsharp.h
            algorithms/lsharp.cpp
            #algorithms/ldot.h
            #algorithms/ldot.cpp
            algorithms/paul.h
            algorithms/paul.cpp

            # PDFAs and WFAs
            algorithms/probabilistic_lsharp.h
            algorithms/probabilistic_lsharp.cpp
            algorithms/weighted_lsharp.h
            algorithms/weighted_lsharp.cpp

            # databases
            databases/database_base.h
            databases/database_headers.h
            databases/prefix_tree_database.h
            databases/prefix_tree_database.cpp

            # oracles
            oracle/oracle_base.h
            oracle/oracle_base.cpp
            oracle/input_file_oracle.h
            oracle/input_file_oracle.cpp
            oracle/discrete_output_sul_oracle.h
            oracle/discrete_output_sul_oracle.cpp
            oracle/string_probability_oracle.h
            oracle/string_probability_oracle.cpp
            #oracle/sqldb_sul_regex_oracle.h
            #oracle/sqldb_sul_regex_oracle.cpp
            #oracle/sqldb_sul_random_oracle.h
            #oracle/sqldb_sul_random_oracle.cpp
            oracle/paul_oracle.h
            oracle/paul_oracle.cpp

            # Counterexample search strategies
            oracle/cex_search_strategies/cex_search_strategy_factory.h
            oracle/cex_search_strategies/cex_search_strategy_factory.cpp
            oracle/cex_search_strategies/search_base.h
            oracle/cex_search_strategies/search_base.cpp
            oracle/cex_search_strategies/breadth_first_search.h
            oracle/cex_search_strategies/breadth_first_search.cpp
            oracle/cex_search_strategies/random_string_search.h
            oracle/cex_search_strategies/random_string_search.cpp
            oracle/cex_search_strategies/random_w_method.h
            oracle/cex_search_strategies/random_w_method.cpp
            oracle/cex_search_strategies/targeted_bfs_walk.h
            oracle/cex_search_strategies/targeted_bfs_walk.cpp
            oracle/cex_search_strategies/fringe_walk.h
            oracle/cex_search_strategies/fringe_walk.cpp

            # conflict search strategies
            oracle/cex_conflict_search/conflict_search_base.h
            oracle/cex_conflict_search/conflict_search_base.cpp
            oracle/cex_conflict_search/linear_conflict_search.h
            oracle/cex_conflict_search/linear_conflict_search.cpp
            oracle/cex_conflict_search/binary_conflict_search.h
            oracle/cex_conflict_search/binary_conflict_search.cpp

            # conflict detectors
            oracle/conflict_detectors/conflict_detector_base.h
            oracle/conflict_detectors/conflict_detector_base.cpp
            oracle/conflict_detectors/type_conflict_detector.h
            oracle/conflict_detectors/type_conflict_detector.cpp
            oracle/conflict_detectors/type_overlap_conflict_detector.h
            oracle/conflict_detectors/type_overlap_conflict_detector.cpp
            oracle/conflict_detectors/string_probability_conflict_detector.h
            oracle/conflict_detectors/string_probability_conflict_detector.cpp

            # Methods to deal with incomplete information
            memory/incomplete_information/ii_base.h
            memory/incomplete_information/ii_base.cpp
            memory/incomplete_information/overlap_fill.h
            memory/incomplete_information/overlap_fill.cpp
            memory/incomplete_information/overlap_fill_batch_wise.h
            memory/incomplete_information/overlap_fill_batch_wise.cpp
            memory/incomplete_information/distinguishing_sequence_fill.h
            memory/incomplete_information/distinguishing_sequence_fill.cpp
            memory/incomplete_information/distinguishing_sequence_fill_fast.h
            memory/incomplete_information/distinguishing_sequence_fill_fast.cpp

            memory/incomplete_information/initializers/ds_initializer_factory.h
            memory/incomplete_information/initializers/ds_initializer_factory.cpp
            memory/incomplete_information/initializers/ds_initializer_registration.h
            memory/incomplete_information/initializers/ds_initializer_registration.cpp
            memory/incomplete_information/initializers/distinguishing_sequences/ds_initializer_base.h
            memory/incomplete_information/initializers/distinguishing_sequences/ds_initializer_pre_generate_sequences.h
            memory/incomplete_information/initializers/distinguishing_sequences/ds_initializer_pre_generate_sequences.cpp
            memory/incomplete_information/initializers/distinguishing_sequences/ds_initializer_collect_from_apta.h
            memory/incomplete_information/initializers/distinguishing_sequences/ds_initializer_collect_from_apta.cpp

            # Everything related to memory and possible memoization techniques
            memory/observation_table.h
            memory/observation_table.cpp
            memory/observation_table_imat.h
            memory/observation_table_imat.cpp
            memory/distinguishing_sequences.h
            memory/distinguishing_sequences.cpp
            memory/adaptive_distinguishing_sequences.h
            memory/adaptive_distinguishing_sequences.cpp
            memory/classification_tree.h
            memory/classification_tree.cpp
            memory/suffix_tree.h
            memory/suffix_tree.cpp

            # The SULs: Interfaces connecting Flexfringe with the system under learning 
            system_under_learning/sul_base.h
            system_under_learning/sul_base.cpp
            system_under_learning/input_file_sul.h
            system_under_learning/input_file_sul.cpp
            system_under_learning/dfa_sul.h
            system_under_learning/dfa_sul.cpp
            system_under_learning/database_sul.h
            system_under_learning/database_sul.cpp
            #system_under_learning/sqldb_sul.h
            #system_under_learning/sqldb_sul.cpp

            # The SULs dedicated to capturing the output of neural networks (or Python scripts having similar outputs)
            system_under_learning/neural_network_suls/nn_sul_base.h
            system_under_learning/neural_network_suls/nn_sul_base.cpp
            system_under_learning/neural_network_suls/nn_binary_output_sul.h
            system_under_learning/neural_network_suls/nn_binary_output_sul.cpp
            system_under_learning/neural_network_suls/nn_discrete_and_float_output_sul.h
            system_under_learning/neural_network_suls/nn_discrete_and_float_output_sul.cpp
            system_under_learning/neural_network_suls/nn_discrete_output_and_hidden_reps_sul.h
            system_under_learning/neural_network_suls/nn_discrete_output_and_hidden_reps_sul.cpp
            system_under_learning/neural_network_suls/nn_discrete_output_sul.h
            system_under_learning/neural_network_suls/nn_discrete_output_sul.cpp
            system_under_learning/neural_network_suls/nn_float_output_sul.h
            system_under_learning/neural_network_suls/nn_float_output_sul.cpp
            system_under_learning/neural_network_suls/nn_float_vector_output_sul.h
            system_under_learning/neural_network_suls/nn_float_vector_output_sul.cpp

            # other functions, such as the parsers for some specific files
            system_under_learning/benchmark_parsers/benchmarkparser_base.h
            system_under_learning/benchmark_parsers/benchmarkparser_base.cpp
            system_under_learning/benchmark_parsers/benchmark_dfaparser.h
            system_under_learning/benchmark_parsers/benchmark_dfaparser.cpp
)


if (ENABLE_PYTHON)
  target_link_libraries(ActiveLearning ${PYTHON_LIBRARIES})
endif()
if (ENABLE_DATABASE)
  target_link_libraries(ActiveLearning libpqxx::pqxx)
endif()
