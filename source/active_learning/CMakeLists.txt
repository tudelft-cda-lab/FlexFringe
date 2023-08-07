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
        "${CMAKE_CURRENT_SOURCE_DIR}/equivalence_oracle/search_strategies"
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

            algorithms/lstar.h
            algorithms/lstar.cpp
            algorithms/lsharp.h
            algorithms/lsharp.cpp

            databases/database_base.h
            databases/database_headers.h
            databases/prefix_tree_database.h
            databases/prefix_tree_database.cpp

            equivalence_oracle/eq_oracle_base.h
            equivalence_oracle/eq_oracle_base.cpp
            equivalence_oracle/input_file_oracle.h
            equivalence_oracle/input_file_oracle.cpp
            equivalence_oracle/active_sul_oracle.h
            equivalence_oracle/active_sul_oracle.cpp

            equivalence_oracle/search_strategies/search_base.h
            equivalence_oracle/search_strategies/search_base.cpp
            equivalence_oracle/search_strategies/breadth_first_search.h
            equivalence_oracle/search_strategies/breadth_first_search.cpp

            memory/observation_table.h
            memory/observation_table.cpp
            memory/adaptive_distinguishing_sequences.h
            memory/adaptive_distinguishing_sequences.cpp 

            system_under_learning/sul_base.h
            system_under_learning/sul_base.cpp
            system_under_learning/input_file_sul.h
            system_under_learning/input_file_sul.cpp
            system_under_learning/dfa_sul.h
            system_under_learning/dfa_sul.cpp
            system_under_learning/database_sul.h
            system_under_learning/database_sul.cpp
            
            system_under_learning/nn_sul_base.h
            system_under_learning/nn_sul_base.cpp
            system_under_learning/nn_sigmoid_sul.h
            system_under_learning/nn_sigmoid_sul.cpp
            
            system_under_learning/benchmark_parsers/benchmarkparser_base.h
            system_under_learning/benchmark_parsers/benchmarkparser_base.cpp
            system_under_learning/benchmark_parsers/benchmark_dfaparser.h
            system_under_learning/benchmark_parsers/benchmark_dfaparser.cpp

            teacher/base_teacher.h
            teacher/base_teacher.cpp
)

target_link_libraries(ActiveLearning ${PYTHON_LIBRARIES})