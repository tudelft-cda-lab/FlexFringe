include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/databases"
        "${CMAKE_CURRENT_SOURCE_DIR}/active_learning_util"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/teacher"
        "${CMAKE_CURRENT_SOURCE_DIR}/system_under_learning"
        "${CMAKE_CURRENT_SOURCE_DIR}/system_under_learning/benchmark_parsers"
        "${CMAKE_CURRENT_SOURCE_DIR}/counterexample_strategies"
        "${CMAKE_CURRENT_SOURCE_DIR}/equivalence_oracle"
        "${CMAKE_CURRENT_SOURCE_DIR}/incomplete_information"
        "${CMAKE_CURRENT_SOURCE_DIR}/equivalence_oracle/cex_search_strategies"
        "${CMAKE_CURRENT_SOURCE_DIR}/equivalence_oracle/cex_conflict_search"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithms"
        )

add_library(ActiveLearning STATIC
            active_learning_main.h
            active_learning_main.cpp

            active_learning_util/definitions.h
            active_learning_util/definitions.cpp
            active_learning_util/common_functions.h
            active_learning_util/common_functions.cpp
            active_learning_util/best_refinement_strategies.h
            active_learning_util/best_refinement_strategies.cpp
            active_learning_util/random_int_generator.h
            active_learning_util/random_int_generator.cpp

            # DFAs
            algorithms/lstar.h
            algorithms/lstar.cpp
            algorithms/lstar_imat.h
            algorithms/lstar_imat.cpp
            algorithms/lsharp.h
            algorithms/lsharp.cpp
            algorithms/ldot.h
            algorithms/ldot.cpp
            algorithms/transformer_l_sharp.h
            algorithms/transformer_l_sharp.cpp
            algorithms/paul.h
            algorithms/paul.cpp

            # PDFAs and WFAs
            algorithms/probabilistic_lsharp.h
            algorithms/probabilistic_lsharp.cpp
            algorithms/weighted_lsharp.h
            algorithms/weighted_lsharp.cpp
            algorithms/transformer_weighted_lsharp.h
            algorithms/transformer_weighted_lsharp.cpp

            databases/database_base.h
            databases/database_headers.h
            databases/prefix_tree_database.h
            databases/prefix_tree_database.cpp

            # Equivalence oracles
            equivalence_oracle/eq_oracle_base.h
            equivalence_oracle/eq_oracle_base.cpp
            equivalence_oracle/input_file_oracle.h
            equivalence_oracle/input_file_oracle.cpp
            equivalence_oracle/active_sul_oracle.h
            equivalence_oracle/active_sul_oracle.cpp
            equivalence_oracle/active_state_sul_oracle.h
            equivalence_oracle/active_state_sul_oracle.cpp
            equivalence_oracle/string_probability_oracle.h
            equivalence_oracle/string_probability_oracle.cpp
            equivalence_oracle/weight_comparing_oracle.h
            equivalence_oracle/weight_comparing_oracle.cpp
            equivalence_oracle/sqldb_sul_regex_oracle.h
            equivalence_oracle/sqldb_sul_regex_oracle.cpp
            equivalence_oracle/sqldb_sul_random_oracle.h
            equivalence_oracle/sqldb_sul_random_oracle.cpp
            equivalence_oracle/paul_oracle.h
            equivalence_oracle/paul_oracle.cpp

            # Counterexample search strategies
            equivalence_oracle/cex_search_strategies/search_base.h
            equivalence_oracle/cex_search_strategies/search_base.cpp
            equivalence_oracle/cex_search_strategies/breadth_first_search.h
            equivalence_oracle/cex_search_strategies/breadth_first_search.cpp
            equivalence_oracle/cex_search_strategies/random_string_search.h
            equivalence_oracle/cex_search_strategies/random_string_search.cpp
            equivalence_oracle/cex_search_strategies/random_w_method.h
            equivalence_oracle/cex_search_strategies/random_w_method.cpp
            equivalence_oracle/cex_search_strategies/targeted_bfs_walk.h
            equivalence_oracle/cex_search_strategies/targeted_bfs_walk.cpp
            equivalence_oracle/cex_search_strategies/fringe_walk.h
            equivalence_oracle/cex_search_strategies/fringe_walk.cpp

            # Conflict search strategies
            equivalence_oracle/cex_conflict_search/dfa_conflict_search_base.h
            equivalence_oracle/cex_conflict_search/dfa_conflict_search_base.cpp
            equivalence_oracle/cex_conflict_search/linear_conflict_search.h
            equivalence_oracle/cex_conflict_search/linear_conflict_search.cpp
            equivalence_oracle/cex_conflict_search/linear_state_query_conflict_search.h
            equivalence_oracle/cex_conflict_search/linear_state_query_conflict_search.cpp
            equivalence_oracle/cex_conflict_search/binary_conflict_search.h
            equivalence_oracle/cex_conflict_search/binary_conflict_search.cpp

            # Methods to deal with incomplete information
            incomplete_information/ii_base.h
            incomplete_information/ii_base.cpp
            incomplete_information/overlap_fill.h
            incomplete_information/overlap_fill.cpp
            incomplete_information/overlap_fill_batch_wise.h
            incomplete_information/overlap_fill_batch_wise.cpp
            incomplete_information/distinguishing_sequence_fill.h
            incomplete_information/distinguishing_sequence_fill.cpp

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

            system_under_learning/sul_base.h
            system_under_learning/sul_base.cpp
            system_under_learning/input_file_sul.h
            system_under_learning/input_file_sul.cpp
            system_under_learning/dfa_sul.h
            system_under_learning/dfa_sul.cpp
            system_under_learning/database_sul.h
            system_under_learning/database_sul.cpp
            system_under_learning/sqldb_sul.h
            system_under_learning/sqldb_sul.cpp

            system_under_learning/nn_sul_base.h
            system_under_learning/nn_sul_base.cpp
            system_under_learning/nn_weighted_output_sul.h
            system_under_learning/nn_weighted_output_sul.cpp

            system_under_learning/benchmark_parsers/benchmarkparser_base.h
            system_under_learning/benchmark_parsers/benchmarkparser_base.cpp
            system_under_learning/benchmark_parsers/benchmark_dfaparser.h
            system_under_learning/benchmark_parsers/benchmark_dfaparser.cpp

            teacher/base_teacher.h
            teacher/base_teacher.cpp
            teacher/teacher_imat.h
            teacher/teacher_imat.cpp
)


if (ENABLE_PYTHON)
  target_link_libraries(ActiveLearning ${PYTHON_LIBRARIES})
endif()
if (ENABLE_DATABASE)
  target_link_libraries(ActiveLearning libpqxx::pqxx)
endif()
