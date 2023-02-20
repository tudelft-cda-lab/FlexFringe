include_directories("${CMAKE_CURRENT_SOURCE_DIR}/memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/system_under_learning"
        "${CMAKE_CURRENT_SOURCE_DIR}/counterexample_strategies"
        "${CMAKE_CURRENT_SOURCE_DIR}/equivalence_oracle"
        "${CMAKE_CURRENT_SOURCE_DIR}/observation_table"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithms"
        )

add_library(ActiveLearning STATIC

            system_under_learning/sul_base.h
            system_under_learning/sul_base.cpp

            equivalence_oracle/eq_oracle_base.h
            equivalence_oracle/eq_oracle_base.cpp

            memory/observation_table.h
            memory/observation_table.cpp
)